import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import random
from typing import Tuple

import subprocess, sys, socket, time
from datetime import datetime  # NEW

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================
# TensorBoard launcher (new CLI: `tensorboard serve`)
# =========================
def _find_free_port(start=6020, max_tries=20):
    for p in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("No free port available")

def launch_tensorboard(logdir: str, start_port: int = 6020, max_tries: int = 20, bind_all: bool = False):
    """
    Launch TensorBoard using the new CLI: `tensorboard serve`.
    """
    port = _find_free_port(start_port, max_tries)
    cmd = [sys.executable, "-m", "tensorboard", "serve", "--logdir", logdir, "--port", str(port)]
    if bind_all:
        cmd.append("--bind_all")
    subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1.0)
    host = "0.0.0.0" if bind_all else "127.0.0.1"
    url = f"http://{host}:{port}"
    print(f"[TENSORBOARD] Live at: {url}")
    print(f"[TENSORBOARD] Logdir: {logdir}")
    return url

# =========================
# CONFIG — tuned for speed
# =========================
CONFIG = {
    "mode": "train",  # "train" or "infer"

    "train": {
        "csv": "/home/aliakhlaghi/codes/all_captions.csv",
        "images_root": "/home/aliakhlaghi/codes/kvasir_vqa_x1/images",
        "image_ext": ".jpg",
        "out_dir": "/home/aliakhlaghi/codes/ckpts",

        "model_id": "Salesforce/instructblip-flan-t5-xl",
        "fallback_model_id": "Salesforce/instructblip-flan-t5-large",

        # Repro
        "seed": 42,

        # Training
        "epochs": 100,
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,
        "max_target_len": 64,

        # Logging cadence (TensorBoard)
        "tb_log_every": 5,

        # Precision / placement
        "prefer_8bit": False,
        "fp16_fallback": True,
        "force_single_gpu": True,
        "use_grad_ckpt": True,

        # LoRA
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,

        # Early stopping
        "early_stop_patience": 3,

        # EVAL cadence
        "eval_every": 5,
        "label_acc_every": 5,     # (not logged)
        "test_every": 5,
        "test_at_end": True,
        "eval_max_batches": None,
        "eval_beams": 1,

        # Periodic snapshots
        "save_every": 5,

        # DataLoader speed knobs
        "num_workers": 8,
        "prefetch_factor": 4,
        "persistent_workers": True,

        # TensorBoard serving options
        "tb_autostart": True,
        "tb_port": 6020,
        "tb_bind_all": False
    },

    "infer": {
        "adapter_dir": "/home/aliakhlaghi/codes/ckpts/best_val",
        "base_model_id": "Salesforce/instructblip-flan-t5-xl",
        "image": "/home/aliakhlaghi/codes/test1.jpg",
        "max_new_tokens_label": 4,
        "max_new_tokens_caption": 32,
        "num_beams": 5,
        "dtype": "auto",
    },
}

# =========================
# Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def derive_polyp_label(text: str) -> str:
    t = (text or "").lower()
    if any(kw in t for kw in ["no polyp", "without polyp", "absent polyp", "no visible polyp"]):
        return "absent"
    if "polyp" in t or "lesion" in t or "adenoma" in t:
        return "present"
    return "absent"


def lora_targets_for_t5() -> list:
    return ["q", "k", "v", "o"]


class CaptionDataset(Dataset):
    """
    Expects CSV: img_id, caption, (optional) polyp_label.
    """
    def __init__(self, df: pd.DataFrame, images_root: str, image_ext: str,
                 processor: InstructBlipProcessor, prompt_template: str):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.image_ext = image_ext
        self.processor = processor
        self.prompt_template = prompt_template

        if "img_id" not in self.df.columns or "caption" not in self.df.columns:
            raise ValueError("CSV must contain 'img_id' and 'caption' columns.")

        self.df["abs_path"] = self.df["img_id"].astype(str).apply(
            lambda x: os.path.join(self.images_root, f"{x}{self.image_ext}")
        )
        missing = [p for p in self.df["abs_path"] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image files not found. First: {missing[0]}")

        if "polyp_label" not in self.df.columns:
            self.df["polyp_label"] = self.df["caption"].apply(derive_polyp_label)

        self.df["polyp_label"] = (
            self.df["polyp_label"].astype(str).str.lower().str.strip().map(
                lambda x: "present" if "present" in x else ("absent" if "absent" in x else x)
            )
        )

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["abs_path"]).convert("RGB")
        prompt = self.prompt_template
        target = f"Polyp: {row['polyp_label']}\nCaption: {row['caption']}"
        gold_label = str(row["polyp_label"]).lower().strip()
        return img, prompt, target, gold_label


def collate_fn(batch, processor: InstructBlipProcessor, max_target_len: int = 64):
    images, prompts, targets, gold_labels = zip(*batch)
    proc = processor(images=list(images), text=list(prompts), return_tensors="pt", padding=True, truncation=True)
    tok = processor.tokenizer(
        list(targets), max_length=max_target_len, padding="max_length", truncation=True, return_tensors="pt"
    )
    labels = tok.input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    proc["labels"] = labels
    proc["gold_labels"] = list(gold_labels)
    return proc


def try_import_bnb():
    try:
        import bitsandbytes as bnb  # noqa: F401
        return True
    except Exception:
        return False


def load_model_quantized_or_fp16(model_id: str, prefer_8bit: bool, fp16_fallback: bool, force_single_gpu: bool):
    from transformers import BitsAndBytesConfig
    device_map = {"": 0} if (torch.cuda.is_available() and force_single_gpu) else "auto"

    if prefer_8bit and try_import_bnb():
        print("[INFO] Loading with 8-bit quantization.")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, device_map=device_map, quantization_config=bnb_config
        )
        return model, "8bit"

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("[INFO] Using bfloat16.")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, device_map=device_map, torch_dtype=torch.bfloat16
        )
        return model, "bf16"

    if fp16_fallback and torch.cuda.is_available():
        print("[INFO] Using float16.")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, device_map=device_map, torch_dtype=torch.float16
        )
        return model, "fp16"

    print("[INFO] Using float32 (CPU or full precision).")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id, device_map=device_map, torch_dtype=torch.float32
    )
    return model, "fp32"


def safe_load_model_with_fallback(args):
    try:
        print(f"[INFO] Loading base: {args['model_id']}")
        return load_model_quantized_or_fp16(
            args["model_id"], args["prefer_8bit"], args["fp16_fallback"], args["force_single_gpu"]
        )
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if args["model_id"] != args["fallback_model_id"]:
            print(f"[WARN] OOM on {args['model_id']}, switching to smaller model: {args['fallback_model_id']}")
            return load_model_quantized_or_fp16(
                args["fallback_model_id"], args["prefer_8bit"], args["fp16_fallback"], args["force_single_gpu"]
            )
        raise


def enforce_schema(text: str) -> Tuple[str, str, str]:
    raw = (text or "").strip()
    m = re.search(r"(?i)polyp\s*:\s*(present|absent)", raw)
    label = m.group(1).lower() if m else "unknown"
    mc = re.search(r"(?i)caption\s*:\s*(.+)", raw)
    cap = (mc.group(1).strip() if mc else "")
    cap = re.sub(r"\s+", " ", cap).strip()
    return label, cap, f"Polyp: {label}\nCaption: {cap if cap else 'N/A'}"


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[v] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ============ TRAIN ============
def cmd_train(args):
    set_seed(args["seed"])
    os.makedirs(args["out_dir"], exist_ok=True)

    # ----- TensorBoard writer: unique run dir + bootstrap -----
    runs_root = os.path.join(args["out_dir"], "logs")
    os.makedirs(runs_root, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logdir_run = os.path.join(runs_root, run_name)
    writer = SummaryWriter(log_dir=logdir_run, flush_secs=2)
    # bootstrap: فعال کردن داشبورد از همان شروع
    writer.add_scalar("Loss/Train", float("nan"), 0)
    writer.flush()
    print(f"[TB] Writing logs to: {logdir_run}")

    # Auto-launch TB on parent so it sees all runs
    if args.get("tb_autostart", False):
        launch_tensorboard(
            logdir=runs_root,  # parent
            start_port=args.get("tb_port", 6020),
            bind_all=args.get("tb_bind_all", False),
        )

    # ---------- DATA: stratified split (70/15/15) ----------
    df = pd.read_csv(args["csv"])
    if "polyp_label" not in df.columns:
        print("[WARN] 'polyp_label' not found in CSV; deriving from caption (noisier).")
        df["polyp_label"] = df["caption"].apply(derive_polyp_label)

    df["polyp_label"] = df["polyp_label"].astype(str).str.lower().str.strip()
    df["polyp_label"] = df["polyp_label"].where(
        df["polyp_label"].isin(["present", "absent"]),
        df["polyp_label"].map(lambda x: "present" if "present" in x else "absent")
    )

    from sklearn.model_selection import train_test_split
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=args["seed"], stratify=df["polyp_label"]
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=args["seed"], stratify=df_temp["polyp_label"]
    )

    df_test.to_csv(os.path.join(args["out_dir"], "test_split.csv"), index=False)
    print(f"[INFO] Dataset = {len(df)} (train={len(df_train)}, val={len(df_val)}, test={len(df_test)})")

    # ---------- MODEL ----------
    processor = InstructBlipProcessor.from_pretrained(args["model_id"])
    model, quant_kind = safe_load_model_with_fallback(args)

    if quant_kind == "8bit":
        model = prepare_model_for_kbit_training(model)

    if args["use_grad_ckpt"] and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args["lora_r"],
        lora_alpha=args["lora_alpha"],
        lora_dropout=args["lora_dropout"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=lora_targets_for_t5()
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    device = next(model.parameters()).device

    # ---------- PROMPT ----------
    prompt_template = (
        "You are a clinical image captioner for colonoscopy frames.\n"
        "Follow this schema exactly:\n"
        "Polyp: present or absent\n"
        "Caption: one short factual sentence (<= 15 words), no speculation.\n"
    )

    # ---------- LOADERS ----------
    num_workers = args["num_workers"]
    prefetch = args["prefetch_factor"]
    persistent = args["persistent_workers"]

    train_loader = DataLoader(
        CaptionDataset(df_train, args["images_root"], args["image_ext"], processor, prompt_template),
        batch_size=args["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch,
        persistent_workers=persistent, drop_last=True,
        collate_fn=lambda b: collate_fn(b, processor, args["max_target_len"])
    )
    val_loader = DataLoader(
        CaptionDataset(df_val, args["images_root"], args["image_ext"], processor, prompt_template),
        batch_size=args["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch,
        persistent_workers=persistent,
        collate_fn=lambda b: collate_fn(b, processor, args["max_target_len"])
    )
    test_loader = DataLoader(
        CaptionDataset(df_test, args["images_root"], args["image_ext"], processor, prompt_template),
        batch_size=args["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch,
        persistent_workers=persistent,
        collate_fn=lambda b: collate_fn(b, processor, args["max_target_len"])
    )

    # ---------- OPTIM/SCHED ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    steps_per_epoch = (len(train_loader) + args["grad_accum"] - 1) // args["grad_accum"]
    total_train_steps = args["epochs"] * steps_per_epoch
    warmup_steps = int(args["warmup_ratio"] * total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)

    # AMP choice
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    use_autocast = (quant_kind in ["8bit", "fp16", "bf16"]) or (amp_dtype == torch.bfloat16)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # ---------- EVAL HELPERS ----------
    def eval_loss(dloader, max_batches=None):
        model.eval()
        total, nb = 0.0, 0
        with torch.no_grad():
            for i, b in enumerate(dloader):
                if (max_batches is not None) and (i >= max_batches):
                    break
                b = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                    outputs = model(**{k: v for k, v in b.items() if isinstance(v, torch.Tensor)})
                    total += outputs.loss.item()
                nb += 1
        model.train()
        return total / max(1, nb)

    def eval_label_accuracy(dloader, max_batches=None, beams=1):
        model.eval()
        acc_sum, nb = 0.0, 0
        with torch.no_grad():
            for i, b in enumerate(dloader):
                if (max_batches is not None) and (i >= max_batches):
                    break
                b = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                gen_kwargs = {k: v for k, v in b.items() if isinstance(v, torch.Tensor) and k not in ("labels",)}
                gen_ids = model.generate(**gen_kwargs, num_beams=beams, do_sample=False, length_penalty=0.0, max_new_tokens=24)
                texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
                preds = [enforce_schema(t)[0] for t in texts]
                golds = b["gold_labels"]
                acc_sum += sum(p == g for p, g in zip(preds, golds)) / len(golds)
                nb += 1
        model.train()
        return acc_sum / max(1, nb)

    # ---------- TRAIN LOOP ----------
    patience = args["early_stop_patience"]
    bad_epochs = 0
    best_val = float("inf")

    for epoch in range(1, args["epochs"] + 1):
        model.train()
        running = 0.0

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Train] epoch {epoch}/{args['epochs']}"):
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                outputs = model(**{k: v for k, v in batch.items() if isinstance(v, torch.Tensor)})
                loss = outputs.loss / args["grad_accum"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args["grad_accum"] == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running += loss.item() * args["grad_accum"]

        avg_train = running / len(train_loader)

        # ---- Eval cadence ----
        do_eval  = (epoch % args["eval_every"] == 0)
        do_test  = (epoch % args.get("test_every", args["eval_every"]) == 0)

        if do_eval:
            avg_val = eval_loss(val_loader, max_batches=args["eval_max_batches"])
        else:
            avg_val = float("nan")

        if do_test:
            test_loss_epoch = eval_loss(test_loader, max_batches=None)
        else:
            test_loss_epoch = float("nan")

        # ---- Save snapshot every N epochs ----
        if args.get("save_every") and (epoch % args["save_every"] == 0):
            snap_dir = os.path.join(args["out_dir"], f"epoch_{epoch:03d}")
            model.save_pretrained(snap_dir)
            processor.save_pretrained(snap_dir)
            print(f"[SNAPSHOT] Saved periodic checkpoint: {snap_dir}")

        # Early stopping on val loss + save best
        if do_eval:
            improved = avg_val < best_val - 1e-6
            if improved:
                best_val = avg_val
                save_dir = os.path.join(args["out_dir"], "best_val")
                model.save_pretrained(save_dir)
                processor.save_pretrained(save_dir)
                bad_epochs = 0
                print(f"[SAVE] New best @ epoch {epoch}: val_loss={best_val:.4f}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[STOP] Early stopping @ epoch {epoch} (best val_loss={best_val:.4f})")
                    # آخرین لاگ قبل از خروج
                    if epoch % args["tb_log_every"] == 0:
                        writer.add_scalar("Loss/Train", avg_train, epoch)
                        if do_eval: writer.add_scalar("Loss/Val", avg_val, epoch)
                        if do_test: writer.add_scalar("Loss/Test", test_loss_epoch, epoch)
                        writer.flush()
                    break

        # ---- Console summary ----
        msg = f"[EPOCH {epoch}] train_loss={avg_train:.4f}"
        if do_eval:  msg += f" | val_loss={avg_val:.4f}"
        if do_test:  msg += f" | test_loss={test_loss_epoch:.4f}"
        print(msg)

        # ---- TensorBoard: only every tb_log_every ----
        if epoch % args["tb_log_every"] == 0:
            writer.add_scalar("Loss/Train", avg_train, epoch)
            if do_eval:
                writer.add_scalar("Loss/Val", avg_val, epoch)
            if do_test:
                writer.add_scalar("Loss/Test", test_loss_epoch, epoch)
            writer.flush()

    # ---- Save 'last' checkpoint ----
    last_dir = os.path.join(args["out_dir"], "last")
    model.save_pretrained(last_dir)
    processor.save_pretrained(last_dir)

    # ---- Final test ----
    if args["test_at_end"]:
        print("[FINAL TEST] evaluating on full test set...")
        final_test_loss = eval_loss(test_loader, max_batches=None)
        print(f"[TEST] loss={final_test_loss:.4f}")
        writer.add_scalar("Loss/Test", final_test_loss, epoch)
        writer.flush()

    writer.close()


# ============ INFERENCE (two-step) ============
def cmd_infer(args):
    processor = InstructBlipProcessor.from_pretrained(args["adapter_dir"])
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    base = InstructBlipForConditionalGeneration.from_pretrained(args["base_model_id"], device_map=device_map)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args["adapter_dir"])
    model.eval()

    device = next(model.parameters()).device
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    img = Image.open(args["image"]).convert("RGB")

    prompt_label = (
        "You are a clinical image captioner for colonoscopy frames.\n"
        "Answer with ONLY one word: present or absent.\n"
        "Is there a polyp visible?"
    )
    inputs_label = processor(images=img, text=prompt_label, return_tensors="pt").to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        gen_label = model.generate(
            **inputs_label, max_new_tokens=args["max_new_tokens_label"], num_beams=args["num_beams"],
            do_sample=False, length_penalty=0.0
        )
    label_text = processor.batch_decode(gen_label, skip_special_tokens=True)[0].strip().lower()
    pred_label = "present" if "present" in label_text else "absent"

    prompt_caption = (
        f"You are a clinical image captioner for colonoscopy frames.\n"
        f"Polyp status: {pred_label}.\n"
        f"Write ONE short factual sentence (<= 15 words), no speculation."
    )
    inputs_caption = processor(images=img, text=prompt_caption, return_tensors="pt").to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        gen_cap = model.generate(
            **inputs_caption, max_new_tokens=CONFIG["infer"]["max_new_tokens_caption"],
            num_beams=CONFIG["infer"]["num_beams"], do_sample=False, length_penalty=0.0
        )
    cap_text = processor.batch_decode(gen_cap, skip_special_tokens=True)[0].strip()
    cap_text = re.sub(r"\s+", " ", cap_text)

    print(f"Polyp: {pred_label}\nCaption: {cap_text}")


# ============ MAIN ============
def main():
    if CONFIG["mode"] == "train":
        cmd_train(CONFIG["train"])
    elif CONFIG["mode"] == "infer":
        cmd_infer(CONFIG["infer"])

if __name__ == "__main__":
    main()

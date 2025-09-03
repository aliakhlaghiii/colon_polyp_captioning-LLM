# -*- coding: utf-8 -*-
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import random
from typing import Tuple, List, Dict, Any

import subprocess, sys, socket, time
from datetime import datetime

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================
# >>> Manual HF token (REQUIRED for gated repos)
# =========================
# NOTE: Replace the string below with your actual HF token.
HF_TOKEN = "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"   # <-- CHANGE THIS


# =========================
# Speed/memory friendly kernels (TF32 + Flash SDPA)
# =========================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    from torch.backends.cuda import sdp_kernel
    # Prefer FlashAttention kernels when available
    sdp_kernel.enable_flash(True)
    sdp_kernel.enable_mem_efficient(True)
    sdp_kernel.enable_math(False)
except Exception:
    pass


# =========================
# TensorBoard launcher
# =========================
def _find_free_port(start=6020, max_tries=20):
    for p in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("No free port available")

def launch_tensorboard(logdir: str, start_port: int = 6020, max_tries: int = 20, bind_all: bool = False):
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
# CONFIG — MedGemma setup (memory-safe tuned)
# =========================
CONFIG = {
    "mode": "train",  # "train" or "infer"

    "train": {
        "csv": "/home/aliakhlaghi/codes/all_captions.csv",
        "images_root": "/home/aliakhlaghi/codes/kvasir_vqa_x1/images",
        "image_ext": ".jpg",
        "out_dir": "/home/aliakhlaghi/codes/ckpts_medgemma",

        # ---- MedGemma base and fallback ----
        "model_id": "google/medgemma-4b-it",
        "fallback_model_id": "google/medgemma-4b-pt",

        # Repro
        "seed": 42,

        # Training (SAFE DEFAULTS for ~16GB GPU)
        "epochs": 20,
        "batch_size": 1,       # ↓↓↓ مهم برای جلوگیری از OOM
        "grad_accum": 8,       # effective batch ~= 8
        "lr": 2e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.06,

        # Logging cadence (TensorBoard)
        "tb_log_every": 1,

        # Precision / placement
        "prefer_8bit": True,
        "fp16_fallback": True,
        "force_single_gpu": True,
        "use_grad_ckpt": True,  # keep True for memory

        # LoRA
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,

        # Early stopping
        "early_stop_patience": 4,

        # EVAL cadence
        "eval_every": 1,
        "label_acc_every": 0,
        "eval_max_batches": 100,   # محدودیت برای صرفه‌جویی VRAM/زمان؛ None = کامل
        "eval_beams": 1,

        # Periodic snapshots
        "save_every": 2,

        # DataLoader knobs
        "num_workers": 4,          # پایین‌تر برای ثبات حافظه
        "prefetch_factor": 2,
        "persistent_workers": True,

        # Vision & text limits
        "image_size": 288,         # کوچک‌تر = VRAM کمتر
        "max_seq_len": 512,        # محدودیت طول سکانس برای جلوگیری از OOM

        # TensorBoard serving options
        "tb_autostart": True,
        "tb_port": 6020,
        "tb_bind_all": False
    },

    "infer": {
        "adapter_dir": "/home/aliakhlaghi/codes/ckpts_medgemma/best_val",
        "base_model_id": "google/medgemma-4b-it",
        "image": "/home/aliakhlaghi/codes/test1.jpg",
        "max_new_tokens_label": 4,
        "max_new_tokens_caption": 32,
        "num_beams": 1,
        "dtype": "auto",
        "image_size": 384,
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

def lora_targets_for_gemma() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class CaptionDataset(Dataset):
    """
    Expects CSV: img_id, caption, (optional) polyp_label.
    Returns messages (system/user/assistant) + gold_label.
    """
    def __init__(self, df: pd.DataFrame, images_root: str, image_ext: str,
                 processor: AutoProcessor, prompt_template: str, image_size: int):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.image_ext = image_ext
        self.processor = processor
        self.prompt_template = prompt_template
        self.image_size = image_size

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

        self.system_msg = (
            "You are a clinical image captioner for colonoscopy frames.\n"
            "Follow this schema exactly:\n"
            "Polyp: present or absent\n"
            "Caption: one short factual sentence (<= 15 words), no speculation."
        )

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["abs_path"]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        target = f"Polyp: {row['polyp_label']}\nCaption: {row['caption']}"
        gold_label = str(row["polyp_label"]).lower().strip()

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_msg}]},
            {"role": "user", "content": [
                {"type": "text", "text": self.prompt_template},
                {"type": "image", "image": img}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]
        return messages, gold_label


def _truncate_batch(input_ids, attention_mask, labels, prompt_lens, max_len: int):
    """
    امن: اگر طول از max_len بیشتر شد، از سمت راست truncate می‌کنیم (سازگار با causal LM).
    prompt_lens را هم clamp می‌کنیم.
    """
    B, L = input_ids.size()
    if L <= max_len:
        return input_ids, attention_mask, labels, [min(pl, L) for pl in prompt_lens]

    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    labels = labels[:, :max_len]
    prompt_lens = [min(pl, max_len) for pl in prompt_lens]
    return input_ids, attention_mask, labels, prompt_lens


def collate_fn_medgemma_train(batch: List, processor: AutoProcessor, max_seq_len: int) -> Dict[str, torch.Tensor]:
    msg_list, gold_labels = zip(*batch)

    # Tokenize full sequence (including assistant text)
    encoded_full = [
        processor.apply_chat_template(
            msgs,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        for msgs in msg_list
    ]

    # Tokenize prompt-only (system+user) to get prefix length for masking
    prompts_only = [[msgs[0], msgs[1]] for msgs in msg_list]
    encoded_prompt = [
        processor.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        for msgs in prompts_only
    ]

    pad_id = processor.tokenizer.pad_token_id
    input_ids = [e["input_ids"].squeeze(0) for e in encoded_full]
    attn = [e["attention_mask"].squeeze(0) for e in encoded_full]
    prompt_lens = [e["input_ids"].shape[-1] for e in encoded_prompt]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)

    labels = input_ids.clone()
    labels[labels == pad_id] = -100
    for i, L in enumerate(prompt_lens):
        labels[i, :L] = -100

    # ---- NEW: hard truncate to max_seq_len to avoid OOM ----
    input_ids, attention_mask, labels, prompt_lens = _truncate_batch(
        input_ids, attention_mask, labels, prompt_lens, max_seq_len
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "gold_labels": list(gold_labels),
    }


def collate_fn_medgemma_gen(batch: List, processor: AutoProcessor, max_seq_len: int):
    prompts = []
    golds = []
    for msgs, g in batch:
        prompts.append([msgs[0], msgs[1]])
        golds.append(g)
    enc = processor.apply_chat_template(
        prompts,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # در جنریشن ورودی را هم اگر خیلی بلند شد truncate می‌کنیم
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    pad_id = processor.tokenizer.pad_token_id

    if input_ids.size(1) > max_seq_len:
        input_ids = input_ids[:, :max_seq_len]
        attention_mask = attention_mask[:, :max_seq_len]
    return {"input_ids": input_ids, "attention_mask": attention_mask}, golds


def try_import_bnb():
    try:
        import bitsandbytes as bnb  # noqa: F401
        return True
    except Exception:
        return False


def load_model_quantized_or_fp16(model_id: str, prefer_8bit: bool, fp16_fallback: bool, force_single_gpu: bool):
    device_map = {"": 0} if (torch.cuda.is_available() and force_single_gpu) else "auto"

    # Prefer 4-bit if possible
    if try_import_bnb():
        print("[INFO] Loading with 4-bit quantization (NF4).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map=device_map, quantization_config=bnb_config, token=HF_TOKEN
        )
        return model, "4bit"

    # Fallback: 8-bit if requested
    if prefer_8bit and try_import_bnb():
        print("[INFO] Loading with 8-bit quantization.")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map=device_map, quantization_config=bnb_config, token=HF_TOKEN
        )
        return model, "8bit"

    # BF16 then FP16 then FP32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("[INFO] Using bfloat16.")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map=device_map, torch_dtype=torch.bfloat16, token=HF_TOKEN
        )
        return model, "bf16"

    if fp16_fallback and torch.cuda.is_available():
        print("[INFO] Using float16.")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map=device_map, torch_dtype=torch.float16, token=HF_TOKEN
        )
        return model, "fp16"

    print("[INFO] Using float32 (CPU or full precision).")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, device_map=device_map, torch_dtype=torch.float32, token=HF_TOKEN
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


def move_to_device(batch: Dict[str, Any], device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ============ TRAIN ============
def cmd_train(args):
    set_seed(args["seed"])
    os.makedirs(args["out_dir"], exist_ok=True)

    runs_root = os.path.join(args["out_dir"], "logs")
    os.makedirs(runs_root, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logdir_run = os.path.join(runs_root, run_name)
    writer = SummaryWriter(log_dir=logdir_run, flush_secs=2)
    writer.add_scalar("Loss/Train", float("nan"), 0)
    writer.flush()
    print(f"[TB] Writing logs to: {logdir_run}")

    if args.get("tb_autostart", False):
        launch_tensorboard(
            logdir=runs_root,
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
    processor = AutoProcessor.from_pretrained(args["model_id"], token=HF_TOKEN)
    # محدودیت طول tokenizer (برای بعضی مدل‌ها اثرگذار است)
    try:
        processor.tokenizer.model_max_length = int(args.get("max_seq_len", 512))
        processor.tokenizer.truncation_side = "right"
    except Exception:
        pass

    model, quant_kind = safe_load_model_with_fallback(args)

    if quant_kind in ["4bit", "8bit"]:
        model = prepare_model_for_kbit_training(model)

    if args["use_grad_ckpt"] and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        # حیاتی: با grad ckpt ناسازگار است
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args["lora_r"],
        lora_alpha=args["lora_alpha"],
        lora_dropout=args["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets_for_gemma()
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    device = next(model.parameters()).device

    # ---------- PROMPT ----------
    prompt_template = (
        "Generate label and caption for this colonoscopy frame.\n"
        "Schema:\nPolyp: present or absent\nCaption: <= 15 words, factual, no speculation."
    )

    # ---------- LOADERS ----------
    num_workers = args["num_workers"]
    prefetch = args["prefetch_factor"]
    persistent = args["persistent_workers"]
    image_size = int(args.get("image_size", 384))
    max_seq_len = int(args.get("max_seq_len", 512))

    train_loader = DataLoader(
        CaptionDataset(df_train, args["images_root"], args["image_ext"], processor, prompt_template, image_size),
        batch_size=args["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch,
        persistent_workers=persistent, drop_last=True,
        collate_fn=lambda b: collate_fn_medgemma_train(b, processor, max_seq_len)
    )
    val_loader_traincollate = DataLoader(
        CaptionDataset(df_val, args["images_root"], args["image_ext"], processor, prompt_template, image_size),
        batch_size=args["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch,
        persistent_workers=persistent,
        collate_fn=lambda b: collate_fn_medgemma_train(b, processor, max_seq_len)
    )
    val_loader_gen = DataLoader(
        CaptionDataset(df_val, args["images_root"], args["image_ext"], processor, prompt_template, image_size),
        batch_size=args["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch,
        persistent_workers=persistent,
        collate_fn=lambda b: collate_fn_medgemma_gen(b, processor, max_seq_len)
    )

    # ---------- OPTIM/SCHED ----------
    try:
        import bitsandbytes as bnb  # noqa: F401
        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(),
            lr=args["lr"],
            weight_decay=args["weight_decay"]
        )
        print("[OPT] Using bitsandbytes PagedAdamW8bit")
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
        print("[OPT] Using torch.optim.AdamW")

    steps_per_epoch = max(1, (len(train_loader) + args["grad_accum"] - 1) // args["grad_accum"])
    total_train_steps = args["epochs"] * steps_per_epoch
    warmup_steps = int(args["warmup_ratio"] * total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)

    # AMP choice
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    use_autocast = True  # فعال بماند؛ با 4bit/8bit یا fp16/bf16 سازگار است
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # ---------- EVAL HELPERS ----------
    def eval_loss(dloader, max_batches=None):
        model.eval()
        total, nb = 0.0, 0
        with torch.no_grad():
            for i, b in enumerate(dloader):
                if (max_batches is not None) and (i >= max_batches):
                    break
                tens = {k: v for k, v in b.items() if isinstance(v, torch.Tensor)}
                tens = move_to_device(tens, device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                    outputs = model(**tens)
                    total += outputs.loss.item()
                nb += 1
        model.train()
        torch.cuda.empty_cache()
        return total / max(1, nb)

    def eval_label_accuracy(dloader_gen, max_batches=None, beams=1):
        model.eval()
        acc_sum, nb = 0.0, 0
        with torch.no_grad():
            for i, (enc, golds) in enumerate(dloader_gen):
                if (max_batches is not None) and (i >= max_batches):
                    break
                enc = move_to_device(enc, device)
                gen_ids = model.generate(
                    **enc, num_beams=beams, do_sample=False, length_penalty=0.0, max_new_tokens=24
                )
                texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
                preds = [enforce_schema(t)[0] for t in texts]
                acc_sum += sum(p == g for p, g in zip(preds, golds)) / len(golds)
                nb += 1
        model.train()
        torch.cuda.empty_cache()
        return acc_sum / max(1, nb)

    # ---------- TRAIN LOOP ----------
    patience = args["early_stop_patience"]
    bad_epochs = 0
    best_val = float("inf")

    for epoch in range(1, args["epochs"] + 1):
        torch.cuda.empty_cache()
        model.train()
        running = 0.0

        prog = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Train] epoch {epoch}/{args['epochs']}")
        for step, batch in prog:
            tens = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
            tens = move_to_device(tens, device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                outputs = model(**tens)
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

            if (step + 1) % 50 == 0:
                # نمایش خلاصه حافظه برای دیباگ OOM های احتمالی
                mem_alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                mem_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
                prog.set_postfix({"loss": f"{running/ (step+1):.4f}", "mem(GB)": f"{mem_alloc:.2f}/{mem_reserved:.2f}"})

        avg_train = running / max(1, len(train_loader))

        do_eval = (epoch % args["eval_every"] == 0)
        if do_eval:
            avg_val = eval_loss(val_loader_traincollate, max_batches=args["eval_max_batches"])
            if args.get("label_acc_every", 0) and (epoch % args["label_acc_every"] == 0):
                acc = eval_label_accuracy(val_loader_gen, max_batches=None, beams=args["eval_beams"])
                print(f"[VAL] label_acc={acc:.4f}")
        else:
            avg_val = float("nan")

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
                    if epoch % args["tb_log_every"] == 0:
                        writer.add_scalar("Loss/Train", avg_train, epoch)
                        writer.add_scalar("Loss/Val", avg_val, epoch)
                        writer.flush()
                    break

        msg = f"[EPOCH {epoch}] train_loss={avg_train:.4f}"
        if do_eval:
            msg += f" | val_loss={avg_val:.4f}"
        print(msg)

        if epoch % args["tb_log_every"] == 0:
            writer.add_scalar("Loss/Train", avg_train, epoch)
            if do_eval:
                writer.add_scalar("Loss/Val", avg_val, epoch)
            writer.flush()

    last_dir = os.path.join(args["out_dir"], "last")
    model.save_pretrained(last_dir)
    processor.save_pretrained(last_dir)
    writer.close()


# ============ INFERENCE ============
def cmd_infer(args):
    processor = AutoProcessor.from_pretrained(args["adapter_dir"])
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    base = AutoModelForImageTextToText.from_pretrained(args["base_model_id"], device_map=device_map, token=HF_TOKEN)

    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args["adapter_dir"])
    model.eval()

    device = next(model.parameters()).device
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    img = Image.open(args["image"]).convert("RGB")
    img = img.resize((int(CONFIG["infer"]["image_size"]), int(CONFIG["infer"]["image_size"])), Image.BICUBIC)

    messages_label = [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are a clinical image captioner for colonoscopy frames."
        )}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Answer with ONLY one word: present or absent. Is there a polyp visible?"},
            {"type": "image", "image": img}
        ]}
    ]
    enc_label = processor.apply_chat_template(
        messages_label, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    ).to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        gen_label = model.generate(
            **enc_label, max_new_tokens=args["max_new_tokens_label"], num_beams=args["num_beams"],
            do_sample=False, length_penalty=0.0
        )
    label_text = processor.batch_decode(gen_label, skip_special_tokens=True)[0].strip().lower()
    pred_label = "present" if "present" in label_text else "absent"

    messages_caption = [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are a clinical image captioner for colonoscopy frames."
        )}]},
        {"role": "user", "content": [
            {"type": "text", "text": (
                f"Polyp status: {pred_label}. Write ONE short factual sentence (<= 15 words), no speculation."
            )},
            {"type": "image", "image": img}
        ]}
    ]
    enc_cap = processor.apply_chat_template(
        messages_caption, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    ).to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        gen_cap = model.generate(
            **enc_cap, max_new_tokens=CONFIG["infer"]["max_new_tokens_caption"],
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
    if not HF_TOKEN or HF_TOKEN.strip() == "" or HF_TOKEN.startswith("hf_put"):
        print("[WARN] HF_TOKEN is not set. Replace HF_TOKEN string at the top of the file with your real token.")
    main()

# ==============================
# ========= EDIT THESE =========
INPUT_DIR      = "/home/aliakhlaghi/codes/test_frame"
OUTPUT_CSV     = "/home/aliakhlaghi/codes/captions_out.csv"
OUTPUT_JSONL   = None  # e.g., "/home/aliakhlaghi/codes/captions_out.jsonl"

CHECKPOINT_DIR = "/home/aliakhlaghi/codes/ckpts/best_val"
BASE_MODEL     = "Salesforce/instructblip-flan-t5-xl"
PROCESSOR_DIR  = None

BATCH_SIZE     = 8
DTYPE          = "float16"
FORCE_CPU      = False
RECURSIVE      = False

# control which images to process (inclusive global indexes)
START_INDEX    = 0
END_INDEX      = 40
# ==============================

import os
import re
import json
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm

import torch
import pandas as pd
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ---------- helpers ----------
def parse_caption(text: str) -> str:
    """
    Extract only the caption. We no longer keep/need the 'Polyp' line.
    Accepts outputs like:
      'Caption: small sessile polyp near fold.'
    or any plain sentence; falls back to cleaned text if needed.
    """
    raw = (text or "").strip()
    m = re.search(r"(?i)caption\s*:\s*(.+)$", raw, flags=re.MULTILINE)
    if m:
        cap = m.group(1).strip()
    else:
        # remove any 'Polyp:' lines if present, then take first non-empty line
        cleaned = re.sub(r"(?i)^\s*polyp\s*:\s*.*$", "", raw, flags=re.MULTILINE).strip()
        cap = ""
        for line in cleaned.splitlines():
            line = line.strip()
            if line:
                cap = line
                break
    cap = re.sub(r"\s+", " ", cap).strip()
    return cap if cap else "N/A"

def make_prompt() -> str:
    # Simplified prompt: only ask for a Caption line
    return (
        "You are a clinical image captioner for colonoscopy frames. "
        "Write a concise, factual, single-sentence caption with NO speculation. "
        "Output MUST strictly follow this schema:\n"
        "Caption: <one short sentence describing key findings>\n"
    )

def torch_dtype_from_str(s):
    if s == "auto":
        return None
    return getattr(torch, s)

def load_processor(processor_dir, checkpoint_dir, fallback_model_id_or_path):
    # Prefer explicit processor dir if provided
    if processor_dir and os.path.isdir(processor_dir):
        try:
            return InstructBlipProcessor.from_pretrained(processor_dir)
        except Exception:
            pass
    # Try the checkpoint (some checkpoints include processor files)
    if os.path.isdir(checkpoint_dir):
        try:
            return InstructBlipProcessor.from_pretrained(checkpoint_dir)
        except Exception:
            pass
    # Fallback to the base model
    return InstructBlipProcessor.from_pretrained(fallback_model_id_or_path)

def is_peft_adapter_dir(path):
    return os.path.exists(os.path.join(path, "adapter_config.json"))

def is_standalone_model_dir(path):
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    files = os.listdir(path)
    return ("pytorch_model.bin" in files) or any(f.endswith(".safetensors") for f in files)

def load_model_standalone(checkpoint_dir, dtype, device):
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if device == "cpu":
        model = InstructBlipForConditionalGeneration.from_pretrained(checkpoint_dir, **kwargs)
        model.to("cpu")
    else:
        model = InstructBlipForConditionalGeneration.from_pretrained(checkpoint_dir, device_map="auto", **kwargs)
    return model

def load_model_with_adapter(base_model_id_or_path, adapter_dir, dtype, device):
    """
    Meta-safe: load base WITHOUT device_map, attach adapter, then move to device.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    base = InstructBlipForConditionalGeneration.from_pretrained(
        base_model_id_or_path,
        low_cpu_mem_usage=False,
        **kwargs
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.to(device)
    return model

# ---- NATURAL (human) SORT: 0,1,2,...,10,11 instead of 0,1,10,100...
def _natural_key_from_path(p: str):
    name = os.path.splitext(os.path.basename(p))[0]
    parts = re.findall(r'\d+|\D+', name)
    return [int(t) if t.isdigit() else t.lower() for t in parts]

def gather_images(root: str, recursive: bool = False) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = []
    if recursive:
        for dp, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(dp, f))
    else:
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
                paths.append(p)
    # natural numeric sort
    paths.sort(key=_natural_key_from_path)
    return paths

# ---------- main ----------
def main():
    # Device & dtype
    device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype_from_str(DTYPE)

    # Processor
    processor = load_processor(PROCESSOR_DIR, CHECKPOINT_DIR, BASE_MODEL)

    # Model
    has_adapter = is_peft_adapter_dir(CHECKPOINT_DIR)
    has_standalone = is_standalone_model_dir(CHECKPOINT_DIR)
    if has_adapter and not has_standalone:
        print("[INFO] Detected PEFT adapter dir. Loading base + adapter (meta-safe).")
        model = load_model_with_adapter(BASE_MODEL, CHECKPOINT_DIR, dtype, device)
    elif has_standalone:
        print("[INFO] Detected standalone model dir.")
        model = load_model_standalone(CHECKPOINT_DIR, dtype, device)
    else:
        try:
            print("[WARN] Type ambiguous. Trying as adapter...")
            model = load_model_with_adapter(BASE_MODEL, CHECKPOINT_DIR, dtype, device)
        except Exception as e:
            print(f"[WARN] Adapter load failed: {e}\nTrying as standalone...")
            model = load_model_standalone(CHECKPOINT_DIR, dtype, device)

    model.eval()

    # Images (get full, globally ordered list)
    all_img_paths = gather_images(INPUT_DIR, recursive=RECURSIVE)
    if not all_img_paths:
        raise RuntimeError(f"No images found in: {INPUT_DIR}")

    # Pair each path with its global sequential index (img_id)
    indexed = list(enumerate(all_img_paths))  # [(0, path0), (1, path1), ...]
    total = len(indexed)

    # Clamp and slice by global indexes (inclusive)
    start = max(0, START_INDEX)
    end_inclusive = min(total - 1, END_INDEX)
    if start > end_inclusive:
        raise RuntimeError(f"Requested slice [{START_INDEX}:{END_INDEX}] is empty for {total} images.")

    subset = indexed[start : end_inclusive + 1]  # list of (img_id, path)
    print(f"[INFO] Processing {len(subset)} images (global indexes {start}..{end_inclusive}).")

    prompt = make_prompt()
    results = []

    # Batching over (img_id, path)
    for i in tqdm(range(0, len(subset), BATCH_SIZE), desc="Captioning"):
        batch = subset[i:i+BATCH_SIZE]

        # Load PIL images (skip broken ones)
        images = []
        ok_ids: List[int] = []
        ok_paths: List[str] = []
        for img_id, p in batch:
            try:
                images.append(Image.open(p).convert("RGB"))
                ok_ids.append(img_id)
                ok_paths.append(p)
            except Exception as e:
                results.append({"img_id": img_id, "caption": f"Image load failed: {e}"})

        if not images:
            continue

        # Prepare inputs
        device_inputs = processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        device_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in device_inputs.items()}

        # Generate
        with torch.no_grad():
            gen_ids = model.generate(
                **device_inputs,
                max_new_tokens=64,
                num_beams=3
            )
        texts = processor.batch_decode(gen_ids, skip_special_tokens=True)

        # Collect ONLY img_id and caption (img_id is the global sequential index)
        for img_id, t in zip(ok_ids, texts):
            cap = parse_caption(t)
            results.append({
                "img_id": img_id,
                "caption": cap
            })

    # Save CSV (ONLY img_id, caption)
    df = pd.DataFrame(results, columns=["img_id", "caption"])
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Wrote {len(df)} rows -> {OUTPUT_CSV}")

    # Optional JSONL (ONLY img_id, caption)
    if OUTPUT_JSONL:
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({"img_id": int(r["img_id"]), "caption": r["caption"]}, ensure_ascii=False) + "\n")
        print(f"[DONE] Wrote JSONL -> {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()

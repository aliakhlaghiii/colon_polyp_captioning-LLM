# universal_infer_captions_batch.py
# -*- coding: utf-8 -*-
"""
Batch captioning for a folder of images using a LoRA-adapted InstructBLIP (flan-t5).
- Edit CONFIG below.
- Uses GPU (CUDA) if available (bf16/float16 when possible).
- Saves a CSV with two columns: filename, caption (sorted numerically by filename).
"""

import os, re, csv
from typing import List, Tuple
from PIL import Image
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from peft import PeftModel

CONFIG = {
    # Paths
    "adapter_dir": "/home/aliakhlaghi/codes/ckpts/best_val",  # LoRA adapter directory
    "base_model_id": "Salesforce/instructblip-flan-t5-xl",            # Base model ID
    "images_dir": "/home/aliakhlaghi/codes/Frame-20250908T134815Z-1-001",         # Folder with 0.jpg, 1.jpg, ...
    "output_csv": "/home/aliakhlaghi/codes/captions_out_for_frames.csv",         # CSV output path

    # Generation
    "num_beams": 5,
    "max_new_tokens_caption": 32,

    # Precision
    "prefer_bf16": True,  # use bf16 if supported
    "allowed_exts": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
}

def _device_and_dtype():
    """Select device (GPU/CPU) and best precision (bf16/float16/float32)."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda and torch.cuda.is_bf16_supported() and CONFIG["prefer_bf16"]:
        return device, torch.bfloat16
    elif use_cuda:
        return device, torch.float16
    else:
        return device, torch.float32

def load_model(adapter_dir: str, base_model_id: str):
    """Load InstructBLIP base model + LoRA adapter and processor."""
    device, amp_dtype = _device_and_dtype()

    # Processor from adapter (ensures same tokenizer/vision config as training)
    processor = InstructBlipProcessor.from_pretrained(adapter_dir)

    # Load base model
    base = InstructBlipForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        torch_dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else None)
    )
    # Merge LoRA
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, processor, device, amp_dtype

def _numeric_key(filename: str) -> Tuple[int, str]:
    """
    Sort filenames numerically (0.jpg, 1.jpg, 10.jpg).
    If no number is found, push to the end but keep deterministic order.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^\s*(\d+)\s*$", base)
    if m:
        return (int(m.group(1)), "")
    else:
        m2 = re.search(r"(\d+)", base)
        if m2:
            return (int(m2.group(1)), base)
        return (10**12, base)

def list_images_sorted(images_dir: str, exts: List[str]) -> List[str]:
    """Return list of image paths sorted numerically by filename."""
    exts = set(e.lower() for e in exts)
    files = []
    for name in os.listdir(images_dir):
        if os.path.splitext(name)[1].lower() in exts:
            files.append(os.path.join(images_dir, name))
    files.sort(key=_numeric_key)
    return files

def generate_caption(model, processor, device, amp_dtype, image_path: str) -> str:
    """Generate one caption for a given image."""
    img = Image.open(image_path).convert("RGB")

    prompt = (
        "You are a clinical image captioner for colonoscopy frames.\n"
        "Output ONLY one short factual sentence (<= 15 words), no speculation or extra text."
    )

    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
        gen_ids = model.generate(
            **inputs,
            num_beams=CONFIG["num_beams"],
            do_sample=False,
            length_penalty=0.0,
            max_new_tokens=CONFIG["max_new_tokens_caption"],
        )

    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    text = re.sub(r"\s+", " ", text)  # clean whitespace
    return text

def write_csv(rows: List[Tuple[str, str]], out_path: str):
    """Write results (filename, caption) into a CSV file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "caption"])
        for r in rows:
            writer.writerow(r)

def main():
    images_dir = CONFIG["images_dir"]
    out_csv = CONFIG["output_csv"]

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = list_images_sorted(images_dir, CONFIG["allowed_exts"])
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {images_dir} with extensions {CONFIG['allowed_exts']}")

    print(f"[INFO] Found {len(image_paths)} images. Loading model...")
    model, processor, device, amp_dtype = load_model(CONFIG["adapter_dir"], CONFIG["base_model_id"])
    print(f"[INFO] Device: {device}, AMP dtype: {amp_dtype}")

    rows: List[Tuple[str, str]] = []
    for idx, img_path in enumerate(image_paths, 1):
        fname = os.path.basename(img_path)
        try:
            caption = generate_caption(model, processor, device, amp_dtype, img_path)
        except Exception as e:
            caption = f"[ERROR] {e.__class__.__name__}: {e}"
        rows.append((fname, caption))
        if idx % 25 == 0 or idx == len(image_paths):
            print(f"[INFO] Processed {idx}/{len(image_paths)}")

    write_csv(rows, out_csv)
    print(f"[DONE] Saved CSV to: {out_csv}")

if __name__ == "__main__":
    main()

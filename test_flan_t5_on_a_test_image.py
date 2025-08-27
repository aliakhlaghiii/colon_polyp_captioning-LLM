# universal_infer_caption.py
# -*- coding: utf-8 -*-
"""
Load LoRA-adapted InstructBLIP and produce a single caption for a given image.
- No argparse; edit CONFIG below.
- Uses GPU (CUDA) if available.
- Outputs only the final caption (single line).
"""

import os, re
from PIL import Image
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from peft import PeftModel

CONFIG = {
    # Paths
    "adapter_dir": "/home/aliakhlaghi/codes/ckpts/best_val",   # e.g., best_val or last
    "base_model_id": "Salesforce/instructblip-flan-t5-xl",
    "image_path": "/home/aliakhlaghi/codes/test1.jpg",

    # Generation
    "num_beams": 5,
    "max_new_tokens_caption": 32,

    # Precision
    "prefer_bf16": True,   # if GPU supports bf16
}

def _device_and_dtype():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda and torch.cuda.is_bf16_supported() and CONFIG["prefer_bf16"]:
        return device, torch.bfloat16
    elif use_cuda:
        return device, torch.float16
    else:
        return device, torch.float32

def load_model(adapter_dir: str, base_model_id: str):
    device, amp_dtype = _device_and_dtype()

    # Load processor from adapter (saves same tokenizer/vision cfg as training)
    processor = InstructBlipProcessor.from_pretrained(adapter_dir)

    # Load base and merge LoRA
    base = InstructBlipForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        torch_dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else None)
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, processor, device, amp_dtype

def generate_caption(model, processor, device, amp_dtype, image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")

    # Prompt: keep it consistent with training but ask for caption only
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
    # Clean to single line
    text = re.sub(r"\s+", " ", text)
    return text

def main():
    m, proc, dev, dtype = load_model(CONFIG["adapter_dir"], CONFIG["base_model_id"])
    cap = generate_caption(m, proc, dev, dtype, CONFIG["image_path"])
    print(cap)

if __name__ == "__main__":
    main()

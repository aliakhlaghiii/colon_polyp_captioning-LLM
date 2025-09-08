# infer/test code
# before runnig this code, we have to export the token in vscode terminal:
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

import os, re                   
from PIL import Image           
import torch                   
from transformers import (      
    AutoProcessor,
    AutoModelForImageTextToText
)
from peft import PeftModel      

# Script-wide configuration
CONFIG = {
    "adapter_dir": "/home/aliakhlaghi/codes/ckpts_medgemma/best_val",  # path to the LoRA folder (best_val)
    "base_model_id": "google/medgemma-4b-it",                          # base model id compatible with the LoRA
    "image_path": "/home/aliakhlaghi/codes/test3.jpg",                 # path to the test image
    "num_beams": 5,                                              
    "max_new_tokens_caption": 32,                                      # max generated tokens
    "prefer_bf16": True,                                             
}

def _device_and_dtype():
    """
    Determine device (GPU/CPU) and an appropriate dtype (bf16/float16/float32).
    """
    use_cuda = torch.cuda.is_available()                               
    device = torch.device("cuda:0" if use_cuda else "cpu")           
    if use_cuda and torch.cuda.is_bf16_supported() and CONFIG["prefer_bf16"]:
        return device, torch.bfloat16                               
    elif use_cuda:
        return device, torch.float16                                  
    else:
        return device, torch.float32                                

def load_model(adapter_dir: str, base_model_id: str):
    """
    Load the Processor, the base model, and attach the LoRA adapter.
    Returns: model, processor, device, amp_dtype
    """
    device, amp_dtype = _device_and_dtype()                        

    hf_token = (os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))

    try:
        processor = AutoProcessor.from_pretrained(adapter_dir, token=hf_token)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model_id, token=hf_token)

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        token=hf_token,                                             
        device_map={"": 0} if torch.cuda.is_available() else "auto",  
        torch_dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else None),
    )

    # Attach the LoRA adapter to the base model
    model = PeftModel.from_pretrained(base, adapter_dir, token=hf_token)
    model.eval()                                                      
    return model, processor, device, amp_dtype

def generate_caption(model, processor, device, amp_dtype, image_path: str) -> str:
    """
    Build the input (text + image) with a chat template, generate a caption, and return a cleaned output.
    """
    img = Image.open(image_path).convert("RGB")                      
    # all the comments below (messages) are changeable
    messages = [
        {"role": "user", "content": [
            {"type": "image"},                                       
            {"type": "text", "text":
             "You are a clinical image captioner for colonoscopy frames. "
             "Output ONLY one short factual sentence (<= 15 words), no speculation or extra text."
            }
        ]}
    ]

    chat_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,  
        tokenize=False,               
    )

    inputs = processor(
        text=[chat_text],         
        images=[img],                 
        return_tensors="pt",         
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}  # move inputs to the device (GPU/CPU)

    # 4) Generate with autocast (enabled only on CUDA)
    use_amp = (device.type == "cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        gen_ids = model.generate(
            **inputs,
            num_beams=CONFIG["num_beams"],            
            do_sample=False,                         
            length_penalty=0.0,                      
            max_new_tokens=CONFIG["max_new_tokens_caption"], 
        )

    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return re.sub(r"\s+", " ", text)                 
# main
def main():
    """
    Entry point: load model/processor, generate a caption, and print it.
    """
    m, proc, dev, dtype = load_model(CONFIG["adapter_dir"], CONFIG["base_model_id"])
    cap = generate_caption(m, proc, dev, dtype, CONFIG["image_path"])
    print(cap)                                       

if __name__ == "__main__":
    main()

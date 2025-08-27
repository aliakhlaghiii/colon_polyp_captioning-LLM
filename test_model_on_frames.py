#!/usr/bin/env python
# test_frames_medclip_retrieval.py
# Use your fine-tuned MedCLIP (.pth) to retrieve the best caption for each frame.
# - Precompute caption embeddings once (cached to disk)
# - Process frames recursively, natural-sorted
# - Print per-frame best caption immediately + latency
# - Save CSV of all results
# - Alert on first polyp-like caption

import os, re, sys, time, warnings
warnings.filterwarnings("ignore", message="`encoder_attention_mask` is deprecated", category=FutureWarning)

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

from medclip import MedCLIPModel, MedCLIPProcessor

# ===================== CONFIG =====================
MODEL_PATH         = "/home/aliakhlaghi/codes/final_trained_models/medclip_finetuned_final_20250811_014842.pth"
FRAMES_FOLDER      = "/home/aliakhlaghi/codes/test_frame"   # folder with frames (subfolders ok)
CAPTION_CSV        = "/home/aliakhlaghi/codes/captions_for_images_based_on_polyp_size.csv"
CAPTION_EMB_CACHE  = "/home/aliakhlaghi/codes/caption_embeds.pt"  # cache file for fast re-runs
OUT_CSV            = "/home/aliakhlaghi/codes/frame_captions_medclip.csv"

BATCH_SIZE_TEXT    = 256   # for first-time caption encoding
MAX_LEN            = 32
TOP_K              = 1     # only the best caption
POLYP_PATTERNS     = [r"\bpolyp\b", r"\badenoma\b", r"\bhyperplastic\b", r"\blesion\b", r"\bgrowth\b", r"\bmass\b"]
ALERT_ON_FIRST_POLYP = True
# ==================================================

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ---- Load MedCLIP model ----
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
print("[INFO] Loading MedCLIP model…")
model = MedCLIPModel().to(device)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()
processor = MedCLIPProcessor()

# ---- Transform for frames ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---- Load captions ----
if not os.path.exists(CAPTION_CSV):
    raise FileNotFoundError(f"Caption file not found: {CAPTION_CSV}")
df = pd.read_csv(CAPTION_CSV)
if "caption" not in df.columns:
    raise ValueError("CSV must contain a 'caption' column.")
df["caption"] = df["caption"].astype(str).str.strip()
df = df[df["caption"] != ""].dropna(subset=["caption"])
captions = df["caption"].tolist()
if not captions:
    raise ValueError("No valid captions found in CSV.")
print(f"[INFO] Captions loaded: {len(captions)}")

# ---- Caption embeddings: load cache or compute once ----
if os.path.exists(CAPTION_EMB_CACHE):
    print(f"[INFO] Loading cached caption embeddings from: {CAPTION_EMB_CACHE}")
    saved = torch.load(CAPTION_EMB_CACHE, map_location=device)
    caption_embs = saved["embeddings"].to(device)
    # (Optional) trust the cached captions order; warn if lengths differ
    if len(saved.get("captions", [])) != len(captions):
        print("[WARN] Cached captions count differs from current CSV. Consider regenerating the cache.")
else:
    print(f"[INFO] Encoding {len(captions)} captions in batches of {BATCH_SIZE_TEXT}… (first run only)")
    text_emb_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), BATCH_SIZE_TEXT), desc="[Text Batches]"):
            batch_caps = captions[i:i+BATCH_SIZE_TEXT]
            tok = processor.tokenizer(
                batch_caps,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            tok = {k: v.to(device) for k, v in tok.items() if k != "token_type_ids"}
            text_emb = model.text_model(**tok)        # [B, D] or [B, L, D]
            if text_emb.dim() == 3:
                text_emb = text_emb[:, 0, :]          # CLS
            text_emb = F.normalize(text_emb, dim=1)    # unit length
            text_emb_list.append(text_emb)
    caption_embs = torch.cat(text_emb_list, dim=0).to(device)  # [N, D]
    # cache for future instant runs
    os.makedirs(os.path.dirname(CAPTION_EMB_CACHE), exist_ok=True)
    torch.save({"embeddings": caption_embs.detach().cpu(), "captions": captions}, CAPTION_EMB_CACHE)
    print(f"[INFO] Saved caption embedding cache → {CAPTION_EMB_CACHE}")

# ---- Collect frames (recursive, case-insensitive) ----
valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
frame_paths = []
for root, _, files in os.walk(FRAMES_FOLDER):
    for f in files:
        if os.path.splitext(f)[1].lower() in valid_ext:
            frame_paths.append(os.path.join(root, f))

if not frame_paths:
    print(f"[ERROR] No image files found under: {FRAMES_FOLDER}")
    sys.exit(1)

def sort_key(p):
    name = os.path.basename(p)
    nums = re.findall(r"\d+", name)
    return (int(nums[-1]) if nums else -1, name.lower())

frame_paths.sort(key=sort_key)
print(f"[INFO] Found {len(frame_paths)} frames (recursive).")

# ---- Helper: polyp detector in caption ----
def contains_polyp(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in POLYP_PATTERNS)

# ---- Iterate frames, print ASAP, log CSV ----
rows = []
first_polyp_announced = False
t_total = time.time()

with torch.no_grad():
    for i, path in enumerate(frame_paths, start=1):
        t0 = time.time()
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"{i:>6}/{len(frame_paths)}  {os.path.basename(path)} → ERROR opening image: {e}")
            rows.append({
                "index": i, "filename": os.path.basename(path), "path": path,
                "caption": "", "similarity": float("nan"), "elapsed_ms": float("nan"), "error": str(e)
            })
            continue

        # image embedding
        image = transform(img).unsqueeze(0).to(device, non_blocking=True)
        img_emb = model.vision_model(image)
        img_emb = img_emb.view(img_emb.size(0), -1)
        img_emb = F.normalize(img_emb, dim=1)  # [1, D]

        # cosine via dot product of normalized vectors
        sims = (img_emb @ caption_embs.T).squeeze(0)           # [N]
        top_vals, top_idx = torch.topk(sims, k=min(TOP_K, sims.numel()), largest=True, sorted=True)
        best_idx = int(top_idx[0].item())
        best_score = float(top_vals[0].item())
        best_caption = captions[best_idx]

        elapsed_ms = (time.time() - t0) * 1000.0

        # Optional one-time alert on first polyp-like caption
        if ALERT_ON_FIRST_POLYP and not first_polyp_announced and contains_polyp(best_caption):
            print(f"🚨 FIRST POLYP @ {i}/{len(frame_paths)} | {os.path.basename(path)} → {best_caption} (cos={best_score:.4f})")
            first_polyp_announced = True

        print(f"{i:>6}/{len(frame_paths)}  {os.path.basename(path)} → {best_caption}  (cos={best_score:.4f}, {elapsed_ms:.1f} ms)")
        sys.stdout.flush()

        rows.append({
            "index": i,
            "filename": os.path.basename(path),
            "path": path,
            "caption": best_caption,
            "similarity": round(best_score, 6),
            "elapsed_ms": round(elapsed_ms, 3)
        })

print(f"[INFO] Done. Total elapsed: {time.time()-t_total:.2f}s")

# ---- Save CSV ----
out_dir = os.path.dirname(OUT_CSV)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"[WRITE] Saved {len(rows)} rows → {OUT_CSV}")

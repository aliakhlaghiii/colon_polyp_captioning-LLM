# vqa_to_all_captions_gpt35_no_cap.py

import time, csv
import pandas as pd
from typing import List, Dict
from openai import OpenAI

API_KEY = "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"
INPUT_CSV = "/home/aliakhlaghi/codes/kvasir_vqa_x1/kvasir_vqa_train_metadata.csv"
OUTPUT_CSV = "/home/aliakhlaghi/codes/all_captions.csv"
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.2
RETRY = 4
SLEEP = 2.5

COLUMN_NAMES = {
    "img": "img_id",
    "q": "question",
    "a": "answer",  
}

CHUNKING = False
CHUNK_SIZE_QA = 50  

SYS_PROMPT = (
    "You will receive multiple question-answer pairs about a single endoscopy image. "
    "Merge them into ONE concise, training-ready caption (<= 30 words). "
    "Constraints: factual, no speculation, no questions, no patient identifiers, "
    "avoid redundancy, resolve contradictions conservatively, reflect uncertainty if present. "
    "If normal/benign, state that clearly."
)

USER_TEMPLATE = """Image info as Q/A lines:
{lines}

Task: Produce ONE concise caption summarizing the essential, clinically relevant facts of this image (single sentence, <= 30 words).
"""

MERGE_TEMPLATE = """We have {n} partial captions about the SAME image:
{lines}

Task: Merge them into ONE superior, concise caption (single sentence, <= 30 words), obeying all constraints from before.
"""

def read_vqa_rows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = [COLUMN_NAMES["img"], COLUMN_NAMES["q"], COLUMN_NAMES["a"]]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[need].dropna().reset_index(drop=True)

def pack_qa_lines(rows: pd.DataFrame) -> str:
    lines: List[str] = []
    qcol, acol = COLUMN_NAMES["q"], COLUMN_NAMES["a"]
    for _, r in rows.iterrows():
        q = str(r[qcol]).replace("\n", " ").strip()
        a = str(r[acol]).replace("\n", " ").strip()
        if q and a:
            lines.append(f"- Q: {q}\n  A: {a}")
    return "\n".join(lines)

def call_gpt(client: OpenAI, model: str, messages, temperature: float) -> str:
    for attempt in range(1, RETRY + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == RETRY:
                raise
            time.sleep(SLEEP * attempt)
    return ""

def caption_from_qa(client: OpenAI, df_group: pd.DataFrame) -> str:
    if not CHUNKING:
        qa_text = USER_TEMPLATE.format(lines=pack_qa_lines(df_group))
        out = call_gpt(
            client, MODEL,
            [{"role": "system", "content": SYS_PROMPT},
             {"role": "user", "content": qa_text}],
            TEMPERATURE
        )
        return " ".join(out.split())

    chunks = []
    for i in range(0, len(df_group), CHUNK_SIZE_QA):
        part = df_group.iloc[i:i+CHUNK_SIZE_QA]
        qa_text = USER_TEMPLATE.format(lines=pack_qa_lines(part))
        cap = call_gpt(
            client, MODEL,
            [{"role": "system", "content": SYS_PROMPT},
             {"role": "user", "content": qa_text}],
            TEMPERATURE
        )
        chunks.append(" ".join(cap.split()))

    if len(chunks) == 1:
        return chunks[0]

    merge_text = MERGE_TEMPLATE.format(
        n=len(chunks),
        lines="\n".join([f"- {c}" for c in chunks])
    )
    final_cap = call_gpt(
        client, MODEL,
        [{"role": "system", "content": SYS_PROMPT},
         {"role": "user", "content": merge_text}],
        TEMPERATURE
    )
    return " ".join(final_cap.split())

def main():
    client = OpenAI(api_key=API_KEY)
    df = read_vqa_rows(INPUT_CSV)

    img_col = COLUMN_NAMES["img"]
    groups: Dict[str, pd.DataFrame] = dict(tuple(df.groupby(img_col)))

    out_rows: List[Dict[str, str]] = []
    total = len(groups)
    for idx, (img_id, g) in enumerate(groups.items(), start=1):
        caption = caption_from_qa(client, g)
        out_rows.append({"img_id": img_id, "caption": caption})
        if idx % 20 == 0 or idx == total:
            print(f"[{idx}/{total}] processed")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img_id", "caption"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved {len(out_rows)} captions to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

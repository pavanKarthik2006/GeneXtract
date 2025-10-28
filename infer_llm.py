from typing import List, Tuple, Dict, Any, Optional
import io
import os

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ===== EDIT THESE =====
MODEL_NAME = "dslim/bert-base-NER"  # change to a fine-tuned checkpoint if available
FILES = [
    r"C:\Users\Pavan\Downloads\NLP Project\NERDatasets\ANAT_ER IOB\train.tsv",
    r"C:\Users\Pavan\Downloads\NLP Project\NERDatasets\ANAT_ER IOB\devel.tsv",
    r"C:\Users\Pavan\Downloads\NLP Project\NERDatasets\ANAT_ER IOB\test.tsv",
]
TOKEN_COL = 0
SKIP_DOCSTART = True     # skip '-DOCSTART-' lines if present
BATCH_SIZE = 16
AGGREGATION = "simple"   # simple | first | average | max | none
DEVICE = "auto"          # "auto" uses CUDA:0 if available, else CPU
TRUNCATION = True        # truncate long inputs if they exceed model max length

_PUNCT_NO_SPACE_BEFORE = set(list(".,!?;:%)]}"))
_PUNCT_NO_SPACE_AFTER  = set(list("([{"))

def detokenize_with_offsets(tokens: List[str], attach_punct: bool = True) -> Tuple[str, List[Tuple[int, int]]]:
    text_parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        add_space_before = bool(text_parts)
        if attach_punct:
            if tok in _PUNCT_NO_SPACE_BEFORE:
                add_space_before = False
            if i > 0 and tokens[i - 1] in _PUNCT_NO_SPACE_AFTER:
                add_space_before = False
        if add_space_before:
            text_parts.append(" ")
            pos += 1
        start = pos
        text_parts.append(tok)
        pos += len(tok)
        end = pos
        offsets.append((start, end))
    return "".join(text_parts), offsets

def read_conll_tokens(path: str, token_col: int = 0, skip_docstart: bool = True) -> List[List[str]]:
    sents: List[List[str]] = []
    cur: List[str] = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split()
            # FIX: check first column, not the whole list
            if skip_docstart and parts and parts == "-DOCSTART-":
                continue
            if token_col >= len(parts):
                raise ValueError(f"TSV line missing token column {token_col}: {line}")
            cur.append(parts[token_col])
    if cur:
        sents.append(cur)
    return sents

def spans_to_bio_from_offsets(tokens: List[str], tok_offsets: List[Tuple[int, int]], spans: List[Dict[str, Any]]) -> List[str]:
    n = len(tokens)
    assigned: List[Optional[Tuple[int, str, float]]] = [None] * n  # (entity_id, label, score)
    # FIX: enumerate gives (idx, span), so use x[13] not x[22]
    spans_sorted = sorted([(i, s) for i, s in enumerate(spans)],
                          key=lambda x: (int(x[13]["start"]), int(x[13]["end"]), -float(x[13].get("score", 0.0))))
    for ent_id, s in spans_sorted:
        start = int(s["start"]); end = int(s["end"])
        label = str(s.get("entity_group") or s.get("label") or s.get("entity") or "O")
        score = float(s.get("score", 0.0))
        if label == "O":
            continue
        inside = []
        for i, (a, b) in enumerate(tok_offsets):
            if a < end and b > start:
                inside.append(i)
        if not inside:
            continue
        for i in inside:
            cur = assigned[i]
            # FIX: compare to cur[19] (score), not cur[23]
            if cur is None or score > cur[19]:
                assigned[i] = (ent_id, label, score)
    tags = ["O"] * n
    for i in range(n):
        if assigned[i] is None:
            continue
        cur_ent, cur_label, _ = assigned[i]
        # FIX: compare previous entity id (index 0), not whole tuple
        prev_same = (i > 0 and assigned[i - 1] is not None and assigned[i - 1] == cur_ent)
        tags[i] = f"I-{cur_label}" if prev_same else f"B-{cur_label}"
    return tags

def pick_device(dev: str = "auto") -> int:
    if dev != "auto":
        try:
            return int(dev)
        except Exception:
            return -1
    return 0 if torch.cuda.is_available() else -1

def main():
    device_index = pick_device(DEVICE)
    aggregation_strategy = None if AGGREGATION == "none" else AGGREGATION

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=aggregation_strategy,
        device=device_index,
    )

    for in_path in FILES:
        if not os.path.exists(in_path):
            print(f"[WARN] File not found, skipping: {in_path}")
            continue
        sents = read_conll_tokens(in_path, token_col=TOKEN_COL, skip_docstart=SKIP_DOCSTART)
        prepared = []
        for tokens in sents:
            text, tok_offsets = detokenize_with_offsets(tokens, attach_punct=True)
            prepared.append((tokens, text, tok_offsets))

        out_path = in_path + ".pred.tsv"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with io.open(out_path, "w", encoding="utf-8") as out:
            for i in range(0, len(prepared), BATCH_SIZE):
                batch = prepared[i:i + BATCH_SIZE]
                # FIX: use b[13] (text), not b[22]
                texts = [b for b in batch]
                results = ner(texts)
                for (tokens, text, tok_offsets), ents in zip(batch, results):
                    spans = []
                    for e in ents:
                        label = e.get("entity_group") or e.get("entity")
                        spans.append({
                            "start": int(e["start"]),
                            "end": int(e["end"]),
                            "label": str(label),
                            "score": float(e.get("score", 0.0)),
                        })
                    tags = spans_to_bio_from_offsets(tokens, tok_offsets, spans)
                    for tok, tag in zip(tokens, tags):
                        out.write(f"{tok}\t{tag}\n")
                    out.write("\n")
        print(f"[OK] Wrote: {out_path}")

if __name__ == "__main__":
    main()

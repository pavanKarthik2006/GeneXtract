from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

# ====== Device
device = 0 if torch.cuda.is_available() else -1



# ====== Load Gene NER Model (JNLPBA)
gene_model = AutoModelForTokenClassification.from_pretrained("biobert-ner-jnlpba")
gene_tokenizer = AutoTokenizer.from_pretrained("biobert-ner-jnlpba")
gene_pipeline = pipeline(
    "ner", model=gene_model, tokenizer=gene_tokenizer,
    aggregation_strategy="simple", device=device
)

# ====== Load text
with open("gene_disease_abstracts.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# ====== Sliding window chunking
def split_into_sliding_windows(text, max_tokens=128, stride=64):
    words = text.split()
    windows = []
    start = 0
    while start < len(words):
        window = words[start:start + max_tokens]
        if not window:
            break
        windows.append(" ".join(window))
        start += (max_tokens - stride)
    return windows

# ====== Merge multi-token entities
def merge_entities(entities, tag):
    merged = []
    buffer = {"word": "", "entity_group": None, "score": []}
    for ent in entities:
        word = ent["word"]
        score = ent["score"]
        if ent["entity_group"] == buffer["entity_group"]:
            if word.startswith("##"):
                buffer["word"] += word[2:]
            else:
                buffer["word"] += " " + word
            buffer["score"].append(score)
        else:
            if buffer["word"]:
                merged.append({
                    "word": buffer["word"].strip(),
                    "type": tag,
                    "score": sum(buffer["score"]) / len(buffer["score"])
                })
            buffer = {"word": word, "entity_group": ent["entity_group"], "score": [score]}
    if buffer["word"]:
        merged.append({
            "word": buffer["word"].strip(),
            "type": tag,
            "score": sum(buffer["score"]) / len(buffer["score"])
        })
    return merged

# ====== Remove duplicates
def deduplicate_entities(entities):
    seen = set()
    unique_entities = []
    for ent in entities:
        key = (ent["word"].lower(), ent["type"])
        if key not in seen:
            unique_entities.append(ent)
            seen.add(key)
    return unique_entities

# ====== Run NER
chunks = split_into_sliding_windows(full_text)
all_entities = []

for chunk in chunks:
    try:
       
        gene_ents = merge_entities(gene_pipeline(chunk), "GENE")
        all_entities.extend(gene_ents)
    except Exception as e:
        print(f"⚠️ Error processing chunk: {e}")

# ====== Group by type
final_entities = deduplicate_entities(all_entities)
entities_by_type = {}
for ent in final_entities:
    etype = ent["type"]
    if etype not in entities_by_type:
        entities_by_type[etype] = []
    entities_by_type[etype].append((ent["word"], ent["score"]))

# ====== Prepare structured text
structured_output = []
for etype, items in entities_by_type.items():
    structured_output.append(f"\n=== {etype} ===")
    for word, score in items:
        structured_output.append(f"{word}  (Confidence: {score:.4f})")

output_text = "\n".join(structured_output)

# Print to console
print(output_text)

# Save to text file
with open("final_entities.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("\n✅ Structured entity report saved to final_entities.txt")

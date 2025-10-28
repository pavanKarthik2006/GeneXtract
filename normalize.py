import re
import mygene
import pandas as pd

# ============================
# Step 1: Load your file
# ============================
input_file = "final_entities.txt"   # replace with your actual path
with open(input_file, "r", encoding="utf-8") as f:
    raw_lines = f.read().splitlines()

# ============================
# Step 2: Extract & Clean Entities
# ============================
def clean_candidate(text):
    # remove parentheses, punctuation except hyphen
    text = re.sub(r"[^A-Za-z0-9\s\-]", " ", text)
    text = text.replace("-", " ")
    tokens = text.split()
    # keep tokens with length >=3
    return [t.upper() for t in tokens if len(t) >= 3]

candidates = []
for line in raw_lines:
    candidates.extend(clean_candidate(line))

candidates = list(set(candidates))  # unique candidates
print(f"✅ Found {len(candidates)} candidate tokens to normalize")

# ============================
# Step 3: Normalize using MyGene.info
# ============================
mg = mygene.MyGeneInfo()
results = mg.querymany(
    candidates,
    scopes="symbol,alias,name",
    species="human",
    fields="symbol,name,entrezgene,ensembl.gene",
    as_dataframe=True
)

# ============================
# Step 4: Save Results
# ============================
results.reset_index(inplace=True)
results.rename(columns={"query": "Original"}, inplace=True)

output_file = "normalized_gene_entities.csv"
results.to_csv(output_file, index=False)

print(f"✅ Normalized gene entities saved to {output_file}")
print(results.head(10))

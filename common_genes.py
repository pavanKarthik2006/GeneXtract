import pandas as pd

# =========================
# Step 1: Load PPI file
# =========================
ppi_file = "ppi_data_with_genes.csv"   # replace with your file path
ppi_df = pd.read_csv(ppi_file)

# Extract all unique genes from both gene1 and gene2 columns
ppi_genes = set(ppi_df["gene1"]).union(set(ppi_df["gene2"]))
print(f"Total unique PPI genes: {len(ppi_genes)}")

# =========================
# Step 2: Load Normalized Genes
# =========================
normalized_file = "normalized_gene_entities.csv"   # replace with your file path
norm_df = pd.read_csv(normalized_file)

# Drop rows without valid gene symbols
norm_genes = set(norm_df["symbol"].dropna().unique())
print(f"Total normalized genes: {len(norm_genes)}")

# =========================
# Step 3: Find Intersection
# =========================
common_genes = sorted(ppi_genes.intersection(norm_genes))
print(f"✅ Found {len(common_genes)} common genes")

# Save to file
output_file = "common_genes.txt"
with open(output_file, "w") as f:
    for g in common_genes:
        f.write(g + "\n")

print(f"Common genes saved to {output_file}")
print("Example list:", common_genes[:20])

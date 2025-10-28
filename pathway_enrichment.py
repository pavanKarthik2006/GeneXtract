import pandas as pd
from scipy.stats import hypergeom

# =========================
# Step 1: Load files
# =========================
expr_file = r"C:\Users\Pavan\Downloads\NLP Project\breast GE.csv"
llm_file = r"common_genes.txt"
pathway_file = r"C:\Users\Pavan\Downloads\NLP Project\pathway association.csv"

expr_df = pd.read_csv(expr_file)
pathway_df = pd.read_csv(pathway_file)

# LLM genes are in a text file, one per line
with open(llm_file, "r") as f:
    llm_genes = set([line.strip() for line in f if line.strip()])

# =========================
# Step 2: Define sets
# =========================
# Assuming breast GE file has a column "sampleID" with gene names
universe = set(expr_df["sampleID"].dropna().unique())  # all genes measured
common_genes = llm_genes.intersection(universe)        # intersection

print(f"Universe size: {len(universe)}")
print(f"LLM gene set size: {len(llm_genes)}")
print(f"Common genes (used for enrichment): {len(common_genes)}")

N = len(universe)  # universe size
n = len(common_genes)  # sample size

# =========================
# Step 3: Hypergeometric test for each pathway
# =========================
results = []
for idx, row in pathway_df.iterrows():
    pathway_genes = set(str(row["hgnc_symbol_ids"]).split(","))  # adjust column name if needed
    K = len(pathway_genes)  # pathway size
    k = len(common_genes.intersection(pathway_genes))  # overlap

    if k > 0:  # only if there’s overlap
        pval = hypergeom.sf(k - 1, N, K, n)  # right-tail p-value
        results.append({
            "pathway_id": row["external_id"],
            "pathway_name": row.get("pathway_name", f"pathway_{idx}"),
            "pathway_size": K,
            "overlap": k,
            "common_genes_in_pathway": ",".join(common_genes.intersection(pathway_genes)),
            "p_value": pval
        })

# =========================
# Step 4: Save results
# =========================
results_df = pd.DataFrame(results).sort_values("p_value")
results_df.to_csv("pathway_enrichment_results.csv", index=False)

print("✅ Pathway enrichment complete. Results saved to pathway_enrichment_results.csv")
print(results_df.head(10))

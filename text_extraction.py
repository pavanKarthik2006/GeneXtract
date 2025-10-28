from Bio import Entrez
import time

Entrez.email = "as23btb0a09@student.nitw.ac.in"  # Replace with your real email

def fetch_gene_disease_abstracts(query, max_results=100):
    print("🔍 Searching PubMed...")
    search_handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results
    )
    search_results = Entrez.read(search_handle)
    search_handle.close()

    id_list = search_results["IdList"]
    print(f"🔗 Found {len(id_list)} articles.")

    abstracts = []
    for i, pubmed_id in enumerate(id_list):
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=pubmed_id,
                rettype="abstract",
                retmode="text"
            )
            abstract = fetch_handle.read().strip()
            fetch_handle.close()
            if abstract:
                abstracts.append(abstract)
                print(f"✅ Abstract {i+1} fetched.")
            else:
                print(f"⚠️ Abstract {i+1} is empty.")
            time.sleep(0.4)  # Respect NCBI rate limits
        except Exception as e:
            print(f"❌ Error fetching abstract {i+1}: {e}")

    return abstracts


# 👇 Example usage
query = '"gene"[Title/Abstract] OR "disease"[Title/Abstract]'
abstracts = fetch_gene_disease_abstracts(query, max_results=100)

# Save to file
with open("gene_disease_abstracts.txt", "w", encoding="utf-8") as f:
    for abs in abstracts:
        f.write(abs + "\n\n")

print("📁 All abstracts saved to gene_disease_abstracts.txt")

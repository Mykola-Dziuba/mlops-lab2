# generate_iab_embeddings.py
# --- imports -------------------------------------------------------------
import os, json, torch
from sentence_transformers import SentenceTransformer

data_dir   = "data"
file_json  = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.json"
file_emb   = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska-Embeddings.json"

model = SentenceTransformer("ipipan/silver-retriever-base-v1.1",
                            device="cuda" if torch.cuda.is_available() else "cpu")

# --- load pages ----------------------------------------------------------
with open(os.path.join(data_dir, file_json), encoding="utf-8") as f:
    pages = json.load(f)

texts = [p["text"] for p in pages]

# --- embed ---------------------------------------------------------------
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True).tolist()

out = [
    {"page_num": p["page_num"], "embedding": emb}
    for p, emb in zip(pages, embeddings)
]

with open(os.path.join(data_dir, file_emb), "w") as f:
    json.dump(out, f, indent=4)

print(f"Saved {len(out)} embeddings → {file_emb}")


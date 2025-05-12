# milvus_insert_iab.py
import os, json
from pymilvus import MilvusClient

data_dir  = "data"
file_txt  = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.json"
file_emb  = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska-Embeddings.json"

COLL = "rag_texts_and_embeddings"
client = MilvusClient(host="localhost", port="19530")

# --- load json files -----------------------------------------------------
with open(os.path.join(data_dir, file_txt),  encoding="utf-8") as f1, \
     open(os.path.join(data_dir, file_emb),  encoding="utf-8") as f2:
    pages = json.load(f1)            # list of {"page_num": n, "text": ...}
    embs  = json.load(f2)            # list of {"page_num": n, "embedding": [...]}

rows = [
    {"text": p["text"], "embedding": e["embedding"]}
    for p, e in zip(pages, embs)
]

# --- insert --------------------------------------------------------------
client.insert(collection_name=COLL, data=rows)

# make the data available for search
client.load_collection(COLL)

print("Inserted", len(rows), "rows into", COLL)


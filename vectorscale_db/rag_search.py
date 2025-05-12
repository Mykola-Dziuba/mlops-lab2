# rag_search.py
# --- imports -------------------------------------------------------------
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(host="localhost", port="19530")
COLL   = "rag_texts_and_embeddings"

model  = SentenceTransformer("ipipan/silver-retriever-base-v1.1")

def search_pages(query: str, top_k: int = 3) -> list[str]:
    """
    Возвращает *top_k* текстовых фрагментов, наиболее близких к запросу.
    """
    emb = model.encode(query).tolist()

    res = client.search(
        collection_name=COLL,
        data=[emb],
        limit=top_k,
        search_params={"metric_type": "L2"},
        output_fields=["text"],
    )

    # res – список результатов по каждому запросу; берём первый
    hits = res[0]
    return [hit.entity.get("text") for hit in hits]


# короткий тест
if __name__ == "__main__":
    for txt in search_pages("Czym jest sztuczna inteligencja", top_k=2):
        print("-" * 40)
        print(txt[:400], "...")


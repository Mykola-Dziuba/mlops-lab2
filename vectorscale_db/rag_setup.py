# rag_setup.py
from pymilvus import FieldSchema, DataType, CollectionSchema, MilvusClient

client = MilvusClient(host="localhost", port="19530")

VECTOR_LENGTH = 768  # Silver Retriever

id_field   = FieldSchema(name="id",  dtype=DataType.INT64,  is_primary=True, description="primary id")
text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096, description="page text")
emb_field  = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                         dim=VECTOR_LENGTH, description="embedded text")

schema = CollectionSchema(
    fields=[id_field, text_field, emb_field],
    auto_id=True,
    enable_dynamic_field=True,
    description="RAG texts collection",
)

COLL = "rag_texts_and_embeddings"

client.create_collection(collection_name=COLL, schema=schema)

# index HNSW + L2
params = client.prepare_index_params()
params.add_index(field_name="embedding", index_type="HNSW",
                 metric_type="L2", params={"M": 4, "efConstruction": 64})

client.create_index(collection_name=COLL, index_params=params)

print("Collections created:", client.list_collections())


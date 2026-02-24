import chromadb

client = chromadb.PersistentClient(
    path="rag_index/chroma_db"
)

print(client.list_collections())
import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from app.vectorstore.qdrant_client import QdrantClientWrapper

@pytest.mark.skip(reason="Requires real Qdrant cloud instance and API key")
def test_qdrant_upsert():
    os.environ["QDRANT_URL"] = "https://your-qdrant-url"
    os.environ["QDRANT_API_KEY"] = "your-qdrant-api-key"
    client = QdrantClientWrapper(collection_name="test_collection")
    embeddings = [[0.1]*768, [0.2]*768]
    metadatas = [{"book_name": "TestBook", "chunk_index": 0}, {"book_name": "TestBook", "chunk_index": 1}]
    client.upsert_chunks(embeddings, metadatas, batch_size=1)
    print("Upserted 2 points to Qdrant (mock test)") 
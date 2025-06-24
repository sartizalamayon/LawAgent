import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from app.embeddings.embedding_model import EmbeddingModel

def test_embedding_model_basic():
    model = EmbeddingModel()
    texts = ["This is a test.", "এটি একটি পরীক্ষা।", "Another test sentence."]
    embeddings = model.embed_texts(texts, batch_size=2)
    assert len(embeddings) == len(texts)
    assert all(isinstance(e, list) for e in embeddings)
    assert all(len(e) == len(embeddings[0]) for e in embeddings), "All embeddings should have the same length"
    print(f"Embedding shape: {len(embeddings)} x {len(embeddings[0])}") 
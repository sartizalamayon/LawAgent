from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class EmbeddingModel:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch = texts[i:i+batch_size]
            batch_embeds = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True).tolist()
            embeddings.extend(batch_embeds)
        return embeddings 
import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm
import numpy as np
import uuid

class QdrantClientWrapper:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        self._ensure_collection()

    def _ensure_collection(self):
        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=768,  # embedding dim for multilingual-e5-base
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def upsert_chunks(self, embeddings: List[List[float]], metadatas: List[Dict], batch_size: int = 64):
        assert len(embeddings) == len(metadatas)
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Upserting to Qdrant"):
            batch_embeds = np.array(embeddings[i:i+batch_size])
            batch_metas = metadatas[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec.tolist(),
                        payload=meta
                    ) for vec, meta in zip(batch_embeds, batch_metas)
                ],
                wait=True
            ) 
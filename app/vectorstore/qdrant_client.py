import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm
import numpy as np
import uuid
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

class QdrantClientWrapper:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        self._check_collection_exists()

    def _check_collection_exists(self):
        # Only connect if collection exists, else raise error
        collections = self.client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            raise RuntimeError(f"Qdrant collection '{self.collection_name}' does not exist. Please create it in the Qdrant UI or CLI.")

    def upsert_chunks(self, embeddings: List[List[float]], metadatas: List[Dict], batch_size: int = 64):
        assert len(embeddings) == len(metadatas)
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Upserting to Qdrant"):
            batch_embeds = np.array(embeddings[i:i+batch_size])
            batch_metas = metadatas[i:i+batch_size]
            # Ensure both 'chunk_text' and 'page_content' are present in every payload
            for meta in batch_metas:
                text = meta.get('chunk_text') or meta.get('page_content')
                if text:
                    meta['chunk_text'] = text
                    meta['page_content'] = text
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

    def delete_by_book_name(self, book_name: str):
        """Delete all points with the given book_name from the collection."""
        flt = Filter(must=[FieldCondition(key="book_name", match=MatchValue(value=book_name))])
        result = self.client.delete(collection_name=self.collection_name, points_selector=flt, wait=True)
        return result

    def create_payload_index(self, field: str, field_type: str = "keyword"):
        """Create a payload index for the given field (e.g., 'book_name')."""
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field,
            field_schema=qmodels.PayloadSchemaType(field_type)
        ) 
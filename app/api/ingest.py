from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import tempfile
from app.parser.pdf_parser import PDFParser
from app.parser.chunking import TextChunker
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.qdrant_client import QdrantClientWrapper

router = APIRouter()
COLLECTION_NAME = 'law_books'

@router.post('/ingest')
async def ingest_books(files: list[UploadFile] = File(...)):
    results = []
    chunker = TextChunker()
    embedder = EmbeddingModel()
    qdrant = QdrantClientWrapper(COLLECTION_NAME)

    for file in files:
        try:
            # Save uploaded file to a temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            parser = PDFParser(tmp_path)
            pages = parser.parse()
            chunks = chunker.chunk(pages)
            texts = [c["chunk_text"] for c in chunks]
            metadatas = [{k: v for k, v in c.items() if k != "chunk_text"} for c in chunks]
            embeddings = embedder.embed_texts(texts)
            qdrant.upsert_chunks(embeddings, metadatas)
            os.remove(tmp_path)
            results.append({"filename": file.filename, "status": "success", "pages": len(pages), "chunks": len(chunks)})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "error": str(e)})
    return JSONResponse(content={"results": results}) 
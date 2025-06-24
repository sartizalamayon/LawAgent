from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, page_dicts: List[Dict]) -> List[Dict]:
        docs = [page['text'] for page in page_dicts]
        metadatas = [{k: v for k, v in page.items() if k != 'text'} for page in page_dicts]
        # Use tqdm to show progress for chunking
        all_chunks = []
        for i, (doc, meta) in enumerate(tqdm(zip(docs, metadatas), total=len(docs), desc="Chunking pages")):
            splits = self.splitter.create_documents([doc], metadatas=[meta])
            for idx, split in enumerate(splits):
                chunk = dict(split.metadata)
                chunk['chunk_text'] = split.page_content
                chunk['chunk_index'] = idx
                all_chunks.append(chunk)
        return all_chunks 
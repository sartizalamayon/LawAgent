import os
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class BookRetriever:
    def __init__(self, collection_name: str = "law_books"):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
        
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        score_threshold: float = 0.7,
        book_filter: Optional[str] = None,
        page_filter: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for relevant chunks based on query
        
        Args:
            query: Search query in any language
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            book_filter: Filter by specific book name (currently disabled due to index requirements)
            page_filter: Filter by specific page number (currently disabled due to index requirements)
            
        Returns:
            List of dictionaries with chunk information and similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
            
            # Note: Filtering by book_name and page_number requires indexes in Qdrant
            # For now, we'll do post-search filtering
            search_filter = None
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k * 3,  # Get more results for post-filtering
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results and apply post-search filtering
            results = []
            for result in search_results:
                # Skip results with missing chunk_text
                chunk_text = result.payload.get('chunk_text')
                if chunk_text is None:
                    continue
                    
                # Apply post-search filtering
                if book_filter and result.payload.get('book_name') != book_filter:
                    continue
                if page_filter is not None and result.payload.get('page_number') != page_filter:
                    continue
                
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'book_name': result.payload.get('book_name'),
                    'page_number': result.payload.get('page_number'),
                    'chunk_index': result.payload.get('chunk_index'),
                    'chunk_text': chunk_text,
                    'file_path': result.payload.get('file_path'),
                    'ingested_at': result.payload.get('ingested_at')
                })
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
            
            logger.info(f"Retrieved {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def search_by_book(self, book_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific book"""
        return self.search(query, top_k=top_k, book_filter=book_name)
    
    def search_by_page(self, book_name: str, page_number: int, query: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific page of a book"""
        return self.search(query, top_k=top_k, book_filter=book_name, page_filter=page_number)
    
    def get_book_chunks(self, book_name: str, limit: int = 100) -> List[Dict]:
        """Get all chunks from a specific book"""
        try:
            # Get all chunks and filter by book name
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )[0]
            
            chunks = []
            for result in results:
                # Skip results with missing chunk_text
                chunk_text = result.payload.get('chunk_text')
                if chunk_text is None:
                    continue
                    
                # Filter by book name
                if result.payload.get('book_name') == book_name:
                    chunks.append({
                        'id': result.id,
                        'book_name': result.payload.get('book_name'),
                        'page_number': result.payload.get('page_number'),
                        'chunk_index': result.payload.get('chunk_index'),
                        'chunk_text': chunk_text,
                        'file_path': result.payload.get('file_path')
                    })
                    
                    if len(chunks) >= limit:
                        break
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting book chunks: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get all chunks
            all_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )[0]
            
            unique_books = set()
            total_chunks = 0
            
            for result in all_results:
                # Only count chunks with valid text
                if result.payload.get('chunk_text') is not None:
                    total_chunks += 1
                    unique_books.add(result.payload.get('book_name'))
            
            return {
                'collection_name': self.collection_name,
                'total_chunks': total_chunks,
                'unique_books': len(unique_books),
                'book_names': list(unique_books),
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """
        Hybrid search combining semantic similarity with keyword matching
        """
        try:
            # Semantic search
            semantic_results = self.search(query, top_k=top_k * 2, score_threshold=0.5)
            
            # Keyword search (simple implementation)
            keyword_results = self._keyword_search(query, top_k=top_k * 2)
            
            # Combine and rerank results
            combined_results = self._combine_results(
                semantic_results, 
                keyword_results, 
                semantic_weight, 
                keyword_weight
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            raise
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Simple keyword search implementation"""
        try:
            # Get all chunks and filter by keywords
            all_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )[0]
            
            query_words = set(query.lower().split())
            scored_results = []
            
            for result in all_results:
                chunk_text = result.payload.get('chunk_text', '')
                if chunk_text is None:
                    continue
                    
                chunk_words = set(chunk_text.lower().split())
                
                # Calculate keyword overlap
                overlap = len(query_words.intersection(chunk_words))
                if overlap > 0:
                    score = overlap / len(query_words)
                    scored_results.append({
                        'id': result.id,
                        'score': score,
                        'book_name': result.payload.get('book_name'),
                        'page_number': result.payload.get('page_number'),
                        'chunk_index': result.payload.get('chunk_index'),
                        'chunk_text': chunk_text,
                        'file_path': result.payload.get('file_path'),
                        'ingested_at': result.payload.get('ingested_at')
                    })
            
            # Sort by score and return top_k
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            return scored_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error during keyword search: {e}")
            return []
    
    def _combine_results(
        self, 
        semantic_results: List[Dict], 
        keyword_results: List[Dict],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict]:
        """Combine and rerank semantic and keyword search results"""
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            result_id = result['id']
            combined[result_id] = {
                **result,
                'combined_score': result['score'] * semantic_weight
            }
        
        # Add keyword results
        for result in keyword_results:
            result_id = result['id']
            if result_id in combined:
                combined[result_id]['combined_score'] += result['score'] * keyword_weight
            else:
                combined[result_id] = {
                    **result,
                    'combined_score': result['score'] * keyword_weight
                }
        
        # Sort by combined score
        combined_list = list(combined.values())
        combined_list.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_list 
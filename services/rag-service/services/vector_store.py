import logging
import asyncio
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
import json
import pickle

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for embeddings and similarity search"""
    
    def __init__(self):
        self.redis_client = None
        self.embedding_model = None
        self.model_name = "all-MiniLM-L6-v2"
        self.embedding_dim = 384
        
    async def initialize(self):
        """Initialize vector store"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=False
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    async def add_document(self, chunk_id: str, content: str, metadata: Dict):
        """Add document chunk to vector store"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(content)
            
            # Prepare document data
            doc_data = {
                "content": content,
                "metadata": metadata,
                "embedding": embedding.tolist()
            }
            
            # Store in Redis
            await self.redis_client.set(
                f"doc:{chunk_id}",
                pickle.dumps(doc_data)
            )
            
            # Add to document index
            document_id = metadata.get("document_id")
            if document_id:
                await self.redis_client.sadd(f"doc_chunks:{document_id}", chunk_id)
            
            logger.debug(f"Added document chunk: {chunk_id}")
            
        except Exception as e:
            logger.error(f"Error adding document {chunk_id}: {str(e)}")
            raise
    
    async def similarity_search(self, query: str, document_ids: List[str], top_k: int = 5) -> List[Dict]:
        """Search for similar document chunks"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Get all chunks for specified documents
            all_chunks = []
            for doc_id in document_ids:
                chunk_ids = await self.redis_client.smembers(f"doc_chunks:{doc_id}")
                for chunk_id in chunk_ids:
                    chunk_data = await self.redis_client.get(f"doc:{chunk_id.decode()}")
                    if chunk_data:
                        doc = pickle.loads(chunk_data)
                        all_chunks.append({
                            "chunk_id": chunk_id.decode(),
                            "content": doc["content"],
                            "metadata": doc["metadata"],
                            "embedding": np.array(doc["embedding"])
                        })
            
            if not all_chunks:
                logger.warning(f"No chunks found for documents: {document_ids}")
                return []
            
            # Calculate similarities
            similarities = []
            for chunk in all_chunks:
                similarity = self._cosine_similarity(query_embedding, chunk["embedding"])
                similarities.append({
                    "document_id": chunk["metadata"]["document_id"],
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "score": float(similarity),
                    "page": chunk["metadata"].get("chunk_index", 0) + 1
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["score"], reverse=True)
            results = similarities[:top_k]
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    async def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a document"""
        try:
            chunk_ids = await self.redis_client.smembers(f"doc_chunks:{document_id}")
            chunks = []
            
            for chunk_id in chunk_ids:
                chunk_data = await self.redis_client.get(f"doc:{chunk_id.decode()}")
                if chunk_data:
                    doc = pickle.loads(chunk_data)
                    chunks.append({
                        "chunk_id": chunk_id.decode(),
                        "content": doc["content"],
                        "metadata": doc["metadata"]
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str):
        """Delete all chunks for a document"""
        try:
            # Get all chunk IDs
            chunk_ids = await self.redis_client.smembers(f"doc_chunks:{document_id}")
            
            # Delete each chunk
            for chunk_id in chunk_ids:
                await self.redis_client.delete(f"doc:{chunk_id.decode()}")
            
            # Delete document index
            await self.redis_client.delete(f"doc_chunks:{document_id}")
            
            logger.info(f"Deleted document: {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise
    
    async def get_stats(self) -> Dict:
        """Get vector store statistics"""
        try:
            # Count total documents
            doc_keys = await self.redis_client.keys("doc_chunks:*")
            total_documents = len(doc_keys)
            
            # Count total chunks
            chunk_keys = await self.redis_client.keys("doc:*")
            total_chunks = len(chunk_keys)
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "embedding_model": self.model_name,
                "embedding_dimension": self.embedding_dim
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Vector store connections closed")
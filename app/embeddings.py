from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import logging
import gc
from functools import lru_cache

logger = logging.getLogger(__name__)

class OptimizedEmbeddingService:
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = "thenlper/gte-small"):
        """Singleton pattern to ensure only one model instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_name: str = "thenlper/gte-small"):
        if self._initialized:
            return
            
        self.model_name = model_name
        self._load_model()
        self.vector_size = self._model.get_sentence_embedding_dimension()
        self._initialized = True
        logger.info(f"Initialized embedding service with model: {model_name}")
    
    def _load_model(self):
        """Lazy load model with memory optimization"""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device='cpu',  # Use CPU to save GPU memory
                cache_folder=None  # Don't cache to disk
            )
            # Optimize model for inference
            self._model.eval()
            # Enable torch optimizations if available
            try:
                import torch
                if hasattr(torch, 'jit') and hasattr(self._model, '_modules'):
                    # Only compile if it's beneficial
                    pass
            except ImportError:
                pass
    
    @lru_cache(maxsize=1000)  # Cache frequently used embeddings
    def _cached_encode(self, text_hash: str, text: str) -> tuple:
        """Cache embeddings for identical texts"""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=1
        )
        return tuple(embedding.astype(np.float32))  # Use float32 to save memory
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text string (sync version)"""
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
        
        text = text.strip()
        if not text:
            raise ValueError("Input cannot be empty after stripping")
        
        # Create hash for caching
        text_hash = str(hash(text))
        
        try:
            embedding_tuple = self._cached_encode(text_hash, text)
            return list(embedding_tuple)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a batch of texts with memory optimization"""
        if not texts:
            return []
        
        # Validate and clean texts
        clean_texts = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(f"All inputs must be strings, got: {type(text)}")
            clean_text = text.strip()
            if not clean_text:
                raise ValueError("Empty text found after stripping")
            clean_texts.append(clean_text)
        
        try:
            # Process in smaller batches to manage memory
            all_embeddings = []
            for i in range(0, len(clean_texts), batch_size):
                batch = clean_texts[i:i + batch_size]
                
                embeddings = self._model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=min(batch_size, len(batch))
                )
                
                # Convert to float32 and list
                batch_embeddings = embeddings.astype(np.float32).tolist()
                all_embeddings.extend(batch_embeddings)
                
                # Force garbage collection for large batches
                if len(batch) > 16:
                    gc.collect()
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Optimized cosine similarity calculation"""
        # Convert to numpy arrays with float32 for memory efficiency
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        
        # Use numpy's optimized dot product
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Handle zero vectors
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def similarity_batch(self, query_embedding: List[float], embeddings: List[List[float]]) -> List[float]:
        """Compute similarities in batch for better performance"""
        query_vec = np.array(query_embedding, dtype=np.float32)
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        
        # Vectorized computation
        dot_products = np.dot(embedding_matrix, query_vec)
        norms = np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(query_vec)
        
        # Handle zero norms
        similarities = np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms!=0)
        
        return similarities.tolist()
    
    def clear_cache(self):
        """Clear the embedding cache to free memory"""
        self._cached_encode.cache_clear()
        gc.collect()
    
    def get_cache_info(self):
        """Get cache statistics"""
        return self._cached_encode.cache_info()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.clear_cache()
        except:
            pass

# Global singleton instance
_embedding_service = None

def get_embedding_service(model_name: str = "thenlper/gte-small") -> OptimizedEmbeddingService:
    """Get the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = OptimizedEmbeddingService(model_name)
    return _embedding_service

# Create the singleton instance that matches your search service usage
embedding_service = get_embedding_service()

# Convenience functions
def get_embedding(text: str) -> List[float]:
    """Get embedding for a text string"""
    service = get_embedding_service()
    return service.generate_embedding(text)

def get_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Get embeddings for multiple texts"""
    service = get_embedding_service()
    return service.generate_embeddings_batch(texts, batch_size)

def compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    return OptimizedEmbeddingService.cosine_similarity(vec1, vec2)
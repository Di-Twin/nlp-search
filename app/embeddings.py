from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
import hashlib
from .cache import cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service with a specific model"""
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedding service with model: {model_name}")
        self.vector_size = 384  # Model's output dimension
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text string"""
        try:
            # Ensure text is a string and not empty
            if not text or not isinstance(text, str):
                raise ValueError("Input must be a non-empty string")
            
            # Clean and normalize the text
            text = text.strip()
            
            # Run the embedding generation in a thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self._encode_text,
                text
            )
            
            # Convert numpy array to list
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        try:
            return self.model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        try:
            # Validate and convert all texts to strings
            validated_texts = []
            for text in texts:
                if not text or not isinstance(text, str):
                    raise ValueError(f"Invalid text input: {text}")
                validated_texts.append(text.strip())
            
            embeddings = self.model.encode(
                validated_texts,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# Global embedding service instance
embedding_service = EmbeddingService()

# Convenience function for getting embeddings
async def get_embedding(text: str) -> List[float]:
    """Get embedding for a text string using the singleton service"""
    return await embedding_service.generate_embedding(text) 
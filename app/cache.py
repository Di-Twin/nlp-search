import redis
import json
import hashlib
from typing import Optional, Any, List, Dict
import logging
from datetime import timedelta
import os
from dotenv import load_dotenv
import asyncio
from .schemas import SearchResult, NutritionInfo

load_dotenv()

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes for query results
EMBEDDING_TTL = int(os.getenv("EMBEDDING_TTL", "86400"))  # 24 hours for embeddings

class SearchResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SearchResult):
            return {
                "id": obj.id,
                "food_name": obj.food_name,
                "description": obj.description,
                "nutrition": obj.nutrition.dict() if obj.nutrition else None,
                "image_url": obj.image_url,
                "scores": obj.scores
            }
        if isinstance(obj, NutritionInfo):
            return obj.dict()
        return super().default(obj)

class CacheService:
    def __init__(self, redis_url: str = None, ttl: int = 300):
        """Initialize Redis cache service
        
        Args:
            redis_url: Redis connection URL
            ttl: Time to live for cache entries in seconds (default: 5 minutes)
        """
        self.redis_url = redis_url or REDIS_URL
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        self.redis_embeddings = redis.from_url(self.redis_url, db=1, decode_responses=False)  # Separate DB for embeddings
        self.ttl = ttl
        self.embedding_ttl = EMBEDDING_TTL

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key from data"""
        if isinstance(data, (list, dict)):
            data = json.dumps(data, sort_keys=True)
        return f"{prefix}:{hashlib.sha256(str(data).encode()).hexdigest()}"

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Generic get method for any cached data"""
        try:
            # Run Redis operation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self.redis.get, key)
            if data:
                parsed_data = json.loads(data)
                # Convert search results back to SearchResult objects
                if isinstance(parsed_data, dict) and "results" in parsed_data:
                    parsed_data["results"] = [
                        SearchResult(**result) if isinstance(result, dict) else result
                        for result in parsed_data["results"]
                    ]
                return parsed_data
            return None
        except Exception as e:
            logger.error(f"Error getting data from cache with key {key}: {str(e)}")
            return None

    async def set(self, key: str, data: Dict[str, Any], ttl: int = None) -> None:
        """Generic set method for any cached data"""
        try:
            cache_ttl = ttl or self.ttl
            json_data = json.dumps(data, cls=SearchResultEncoder)
            
            # Run Redis operation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                lambda: self.redis.setex(key, cache_ttl, json_data)
            )
            logger.debug(f"Cached data with key {key} for {cache_ttl} seconds")
        except Exception as e:
            logger.error(f"Error setting data in cache with key {key}: {str(e)}")

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        try:
            key = self._generate_key("embedding", text)
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self.redis_embeddings.get, key)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Error getting embedding from cache: {str(e)}")
            return None

    async def set_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text"""
        try:
            key = self._generate_key("embedding", text)
            json_data = json.dumps(embedding)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.redis_embeddings.setex(key, self.embedding_ttl, json_data.encode('utf-8'))
            )
            logger.debug(f"Cached embedding for text: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error setting embedding in cache: {str(e)}")

    async def get_search_results(self, query: str, limit: int) -> Optional[Dict[str, Any]]:
        """Get cached search results"""
        try:
            key = self._generate_key("search", f"{query}:{limit}")
            return await self.get(key)
        except Exception as e:
            logger.error(f"Error getting search results from cache: {str(e)}")
            return None

    async def set_search_results(self, query: str, limit: int, results: Dict[str, Any]) -> None:
        """Cache search results"""
        try:
            key = self._generate_key("search", f"{query}:{limit}")
            await self.set(key, results, self.ttl)
        except Exception as e:
            logger.error(f"Error setting search results in cache: {str(e)}")

    async def invalidate_search_cache(self) -> None:
        """Invalidate all search results cache"""
        try:
            loop = asyncio.get_event_loop()
            keys = await loop.run_in_executor(None, self.redis.keys, "search:*")
            if keys:
                await loop.run_in_executor(None, self.redis.delete, *keys)
                logger.info(f"Invalidated {len(keys)} search cache entries")
        except Exception as e:
            logger.error(f"Error invalidating search cache: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """Check cache service health"""
        try:
            loop = asyncio.get_event_loop()
            # Test connection with a simple ping
            await loop.run_in_executor(None, self.redis.ping)
            
            # Get some stats
            info = await loop.run_in_executor(None, self.redis.info)
            
            return {
                "status": "healthy",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "uptime": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def get_query_cache_sync(self, key: str) -> Optional[Dict[str, Any]]:
        """Synchronous version for backwards compatibility"""
        try:
            data = self.redis.get(f"cache:query:{key}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting query cache: {str(e)}")
            return None

    def set_query_cache_sync(self, key: str, data: Dict[str, Any]) -> None:
        """Synchronous version for backwards compatibility"""
        try:
            self.redis.setex(
                f"cache:query:{key}",
                self.ttl,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.error(f"Error setting query cache: {str(e)}")

# Global cache service instance
_cache_service = None

def get_cache_service() -> CacheService:
    """Get cache service instance (singleton)"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service

# Create the singleton instance that matches your search service usage
cache_service = get_cache_service()

# Legacy RedisCache class for backwards compatibility
class RedisCache:
    def __init__(self):
        self.redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        self.redis_embeddings = redis.Redis.from_url(REDIS_URL, db=1)  # Separate DB for embeddings

    def get_query_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached query results"""
        data = self.redis.get(f"cache:query:{key}")
        if data:
            return json.loads(data)
        return None

    def set_query_cache(self, key: str, data: Dict[str, Any]) -> None:
        """Cache query results"""
        self.redis.setex(
            f"cache:query:{key}",
            CACHE_TTL,
            json.dumps(data, default=str)
        )

    def get_embedding_cache(self, key: str) -> Optional[List[float]]:
        """Get cached embedding vector"""
        data = self.redis_embeddings.get(f"embed:query:{key}")
        if data:
            return json.loads(data)
        return None

    def set_embedding_cache(self, key: str, embedding: List[float]) -> None:
        """Cache embedding vector"""
        self.redis_embeddings.setex(
            f"embed:query:{key}",
            EMBEDDING_TTL,
            json.dumps(embedding)
        )

    def get_suggestions(self, prefix: str, limit: int = 10) -> List[str]:
        """Get autocomplete suggestions"""
        suggestions = self.redis.zrangebylex(
            "suggest:prefix",
            f"[{prefix}",
            f"[{prefix}\xff",
            start=0,
            num=limit
        )
        return suggestions

    def add_suggestion(self, term: str) -> None:
        """Add a term to the suggestion index"""
        self.redis.zadd("suggest:prefix", {term: 0})

# Global cache instance for backwards compatibility
cache = RedisCache()
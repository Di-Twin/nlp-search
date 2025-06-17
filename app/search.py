from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import select, func, text, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from .models import FoodNutrition
from .embeddings import get_embedding_service
from .cache import cache_service
from .schemas import SearchResult, NutritionInfo
import time
import logging
import asyncio
import hashlib
from functools import lru_cache
import gc

logger = logging.getLogger(__name__)

class UltraOptimizedSearchService:
    """Ultra-fast, memory-efficient search service"""
    
    __slots__ = ('_session_factory', '_embedding_service', '_tsquery_cache', '_cache_hits', '_cache_misses')
    
    def __init__(self):
        from .database import async_session
        self._session_factory = async_session
        self._embedding_service = get_embedding_service()
        self._tsquery_cache = {}  # Minimal cache for tsquery
        self._cache_hits = 0
        self._cache_misses = 0

    @lru_cache(maxsize=64)  # Small cache for common queries
    def _build_tsquery(self, query: str) -> str:
        """Build PostgreSQL tsquery with minimal processing"""
        words = query.lower().split()
        if not words:
            return ""
        
        if len(words) == 1:
            word = words[0].replace("'", "")
            return f"{word}:*" if len(word) > 2 else word
        
        # Simple AND query for multiple words
        clean_words = [w.replace("'", "") for w in words if len(w) > 1]
        return " & ".join(clean_words[:4])  # Limit to 4 words max

    @staticmethod
    def _cache_key(query: str, limit: int) -> str:
        """Ultra-fast cache key generation"""
        return f"{hash(query.lower())}:{limit}"

    async def _get_embedding_fast(self, query: str) -> Optional[List[float]]:
        """Fast embedding retrieval with fire-and-forget caching"""
        try:
            # Try cache first (fastest path)
            cached = await cache_service.get_embedding(query)
            if cached:
                self._cache_hits += 1
                return cached
            
            self._cache_misses += 1
            
            # Generate embedding synchronously (faster than async for single queries)
            embedding = self._embedding_service.generate_embedding(query)
            
            # Fire-and-forget cache storage
            if embedding:
                asyncio.create_task(cache_service.set_embedding(query, embedding))
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _build_hybrid_query(self, query: str, embedding: List[float], limit: int):
        """Build optimized hybrid search query"""
        from pgvector.sqlalchemy import Vector
        
        # Semantic similarity
        semantic_dist = func.cosine_distance(
            FoodNutrition.embedding,
            func.cast(embedding, Vector)
        )
        
        # Text similarity (simplified)
        name_sim = func.similarity(FoodNutrition.food_name, query)
        
        # Full-text search
        tsquery_str = self._build_tsquery(query)
        if tsquery_str:
            ts_query = func.plainto_tsquery('english', query)
            text_match = FoodNutrition.document.op('@@')(ts_query)
            ts_rank = func.ts_rank_cd(FoodNutrition.document, ts_query)
        else:
            text_match = func.literal(False)
            ts_rank = func.literal(0.0)
        
        # Optimized WHERE conditions (tighter thresholds for better performance)
        conditions = or_(
            semantic_dist < 0.6,  # Stricter semantic threshold
            name_sim > 0.3,       # Higher name similarity threshold
            text_match
        )
        
        # Simplified scoring (fewer calculations)
        final_score = (
            semantic_dist * 0.5 +           # Semantic weight
            (1.0 - name_sim) * 0.3 +        # Name similarity weight
            (1.0 - ts_rank) * 0.2           # Text search weight
        )
        
        return (
            select(
                FoodNutrition,
                semantic_dist.label('sem_dist'),
                name_sim.label('name_sim'),
                ts_rank.label('text_rank'),
                final_score.label('score')
            )
            .where(conditions)
            .order_by(final_score.asc(), FoodNutrition.id.asc())
            .limit(limit)
        )

    def _build_text_query(self, query: str, limit: int):
        """Build text-only search query"""
        name_sim = func.similarity(FoodNutrition.food_name, query)
        
        tsquery_str = self._build_tsquery(query)
        if tsquery_str:
            ts_query = func.plainto_tsquery('english', query)
            text_match = FoodNutrition.document.op('@@')(ts_query)
            ts_rank = func.ts_rank_cd(FoodNutrition.document, ts_query)
            
            conditions = or_(name_sim > 0.2, text_match)
            order_expr = (ts_rank + name_sim * 2).desc()  # Boost name similarity
        else:
            conditions = name_sim > 0.2
            ts_rank = func.literal(0.0)
            order_expr = name_sim.desc()
        
        return (
            select(
                FoodNutrition,
                name_sim.label('name_sim'),
                ts_rank.label('text_rank')
            )
            .where(conditions)
            .order_by(order_expr, FoodNutrition.id.asc())
            .limit(limit)
        )

    @staticmethod
    def _fast_relevance_score(query_lower: str, name_lower: str, semantic_score: float = 0.0) -> float:
        """Ultra-fast relevance scoring with minimal string operations"""
        # Exact match
        if query_lower == name_lower:
            return 1.0
        
        # Substring matches (optimized with 'in' operator)
        if query_lower in name_lower:
            return 0.9 - (len(name_lower) - len(query_lower)) * 0.01  # Penalize longer names slightly
        
        if name_lower in query_lower:
            return 0.8
        
        # Use semantic or default
        return max(semantic_score, 0.1)

    def _create_result_fast(self, item: FoodNutrition, relevance: float) -> Dict[str, Any]:
        """Create result dict directly (faster than Pydantic model)"""
        return {
            "id": item.id,
            "food_name": item.food_name,
            "description": item.description,
            "nutrition": {
                "energy_kcal": item.energy_kcal or 0,
                "protein_g": item.protein_g or 0,
                "fat_g": item.fat_g or 0,
                "carbohydrates_g": item.carbohydrates_g or 0,
                "fiber_g": item.fiber_g or 0,
                "minerals": {
                    "calcium_mg": item.calcium_mg or 0,
                    "iron_mg": item.iron_mg or 0,
                    "magnesium_mg": item.magnesium_mg or 0,
                    "phosphorus_mg": item.phosphorus_mg or 0,
                    "potassium_mg": item.potassium_mg or 0,
                    "sodium_mg": item.sodium_mg or 0,
                    "zinc_mg": item.zinc_mg or 0
                },
                "vitamins": {
                    "vitamin_c_mg": item.vitamin_c_mg or 0,
                    "thiamin_mg": item.thiamin_mg or 0,
                    "riboflavin_mg": item.riboflavin_mg or 0,
                    "niacin_mg": item.niacin_mg or 0,
                    "vitamin_b6_mg": item.vitamin_b6_mg or 0,
                    "folate_ug": item.folate_ug or 0,
                    "vitamin_a_ug": item.vitamin_a_ug or 0,
                    "vitamin_e_mg": item.vitamin_e_mg or 0,
                    "vitamin_d_ug": item.vitamin_d_ug or 0
                }
            },
            "image_url": item.image_url,
            "relevance": round(relevance, 3)  # Single score field
        }

    async def search(self, query: str, limit: int = 10, use_cache: bool = True) -> Dict[str, Any]:
        """Ultra-optimized search with maximum speed and minimal memory usage"""
        start_time = time.perf_counter()
        
        # Fast validation
        if not query or len(query.strip()) == 0:
            return {"results": [], "meta": {"error": "Empty query"}}
        
        if not 1 <= limit <= 20:  # Reasonable limits
            limit = min(max(limit, 1), 20)
        
        query = query.strip()
        query_lower = query.lower()
        
        # Cache check (fastest path)
        if use_cache:
            cache_key = self._cache_key(query, limit)
            try:
                cached = await cache_service.get_search_results(query, limit)
                if cached:
                    cached["meta"]["cache_hit"] = True
                    cached["meta"]["duration_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
                    return cached
            except Exception:
                pass
        
        try:
            async with self._session_factory() as session:
                # Get embedding (non-blocking)
                embedding = await self._get_embedding_fast(query)
                has_embedding = embedding is not None
                
                # Build query based on embedding availability
                if has_embedding:
                    search_query = self._build_hybrid_query(query, embedding, limit + 5)  # Get a few extra for dedup
                else:
                    search_query = self._build_text_query(query, limit + 5)
                
                # Execute query
                result = await session.execute(search_query)
                rows = result.all()
                
                # Process results with minimal allocations
                results = []
                seen_names = set()
                
                for row in rows:
                    item = row[0]
                    
                    # Fast deduplication
                    name_key = item.food_name.lower()
                    if name_key in seen_names:
                        continue
                    seen_names.add(name_key)
                    
                    # Calculate relevance
                    if has_embedding:
                        semantic_score = max(0.0, 1.0 - float(row.sem_dist)) if hasattr(row, 'sem_dist') else 0.0
                    else:
                        semantic_score = 0.0
                    
                    relevance = self._fast_relevance_score(query_lower, name_key, semantic_score)
                    
                    results.append(self._create_result_fast(item, relevance))
                    
                    if len(results) >= limit:
                        break
                
                # Sort by relevance (usually already sorted, but ensure consistency)
                if len(results) > 1:
                    results.sort(key=lambda x: x["relevance"], reverse=True)
                
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                
                response = {
                    "results": results,
                    "meta": {
                        "duration_ms": duration_ms,
                        "cache_hit": False,
                        "total_results": len(results),
                        "embedding_used": has_embedding,
                        "cache_stats": f"{self._cache_hits}/{self._cache_hits + self._cache_misses}"
                    }
                }
                
                # Fire-and-forget cache storage
                if use_cache and results:
                    asyncio.create_task(cache_service.set_search_results(query, limit, response))
                
                return response
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "results": [],
                "meta": {
                    "error": str(e),
                    "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)
                }
            }

    async def search_by_nutrition(self, filters: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        """Optimized nutrition search with minimal overhead"""
        start_time = time.perf_counter()
        
        if not filters or limit < 1:
            return {"results": [], "meta": {"error": "Invalid parameters"}}
        
        try:
            async with self._session_factory() as session:
                query = select(FoodNutrition)
                
                # Apply filters efficiently (validate attribute exists)
                valid_attrs = {
                    'energy_kcal', 'protein_g', 'fat_g', 'carbohydrates_g', 'fiber_g',
                    'calcium_mg', 'iron_mg', 'magnesium_mg', 'phosphorus_mg', 
                    'potassium_mg', 'sodium_mg', 'zinc_mg', 'vitamin_c_mg'
                }
                
                for key, value in filters.items():
                    if key not in valid_attrs or value is None:
                        continue
                    
                    attr = getattr(FoodNutrition, key)
                    
                    if isinstance(value, dict):
                        if "min" in value and value["min"] is not None:
                            query = query.where(attr >= value["min"])
                        if "max" in value and value["max"] is not None:
                            query = query.where(attr <= value["max"])
                    elif isinstance(value, (int, float)):
                        query = query.where(attr >= value)
                
                # Order by energy for consistency
                query = query.order_by(FoodNutrition.energy_kcal.desc()).limit(min(limit, 20))
                
                result = await session.execute(query)
                items = result.scalars().all()
                
                # Create results efficiently
                results = [self._create_result_fast(item, 1.0) for item in items]
                
                return {
                    "results": results,
                    "meta": {
                        "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
                        "total_results": len(results)
                    }
                }
                
        except Exception as e:
            logger.error(f"Nutrition search error: {e}")
            return {
                "results": [],
                "meta": {
                    "error": str(e),
                    "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)
                }
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        cache_info = self._build_tsquery.cache_info()
        return {
            "tsquery_cache": cache_info._asdict(),
            "embedding_cache_hits": self._cache_hits,
            "embedding_cache_misses": self._cache_misses,
            "hit_ratio": self._cache_hits / max(self._cache_hits + self._cache_misses, 1)
        }

    def clear_caches(self):
        """Clear all caches to free memory"""
        self._build_tsquery.cache_clear()
        self._cache_hits = 0
        self._cache_misses = 0
        gc.collect()

# Global singleton instance
search_service = UltraOptimizedSearchService()

async def init_search_service():
    """Initialize search service (minimal setup)"""
    logger.info("Ultra-optimized search service initialized")

# Convenience function for direct search
async def quick_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """Direct search function for performance-critical paths"""
    return await search_service.search(query, limit, use_cache=True)
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import select, func, text, case, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from .models import FoodNutrition
from .embeddings import embedding_service
from .cache import cache_service  # Updated import
from .schemas import SearchResult, NutritionInfo
import time
import logging
from .database import async_session
import numpy as np
from pgvector.sqlalchemy import Vector
import asyncio
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self._session_factory = async_session
        self._connection_pool = None
        
    async def initialize(self):
        """Initialize the search service"""
        # Pre-warm the connection pool if needed
        logger.info("Search service initialized")

    @lru_cache(maxsize=1000)
    def _create_tsquery(self, query: str) -> str:
        """Create a cached tsquery string from the input query with better typo handling"""
        # Clean the query and create tsquery format
        cleaned_words = [word.strip() for word in query.split() if word.strip()]
        if not cleaned_words:
            return ""
        
        # For single word queries, add prefix matching for typos
        if len(cleaned_words) == 1 and len(cleaned_words[0]) > 2:
            word = cleaned_words[0].replace("'", "''")
            # Add both exact and prefix matching
            return f"{word} | {word}:*"
        
        # For multiple words, escape and join with &
        escaped_words = [word.replace("'", "''") for word in cleaned_words]
        return ' & '.join(escaped_words)

    def _generate_cache_key(self, query: str, limit: int) -> str:
        """Generate a cache key for the search query - matches cache service format"""
        key_string = f"search:{query.lower().strip()}:{limit}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def _get_embedding_with_fallback(self, query: str) -> Optional[List[float]]:
        """Get embedding with error handling and fallback"""
        try:
            # First try to get from cache
            cached_embedding = await cache_service.get_embedding(query)
            if cached_embedding:
                logger.debug(f"Using cached embedding for query: {query}")
                return cached_embedding
            
            # Generate new embedding
            embedding = await embedding_service.generate_embedding(query)
            if embedding:
                # Cache the embedding
                await cache_service.set_embedding(query, embedding)
                logger.debug(f"Generated and cached new embedding for query: {query}")
            
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding for query '{query}': {e}")
            return None

    async def _execute_hybrid_search(
        self, 
        session: AsyncSession, 
        query: str, 
        query_embedding: Optional[List[float]], 
        limit: int
    ) -> List[FoodNutrition]:
        """Execute the hybrid search query with optimized performance and deduplication"""
        
        # Create tsquery with better handling for misspellings
        tsquery = self._create_tsquery(query)
        
        # Enhanced fuzzy matching for better typo handling
        fuzzy_threshold = 0.1  # Lower threshold for better fuzzy matching
        
        if query_embedding:
            # Hybrid search with improved scoring
            semantic_distance_expr = func.cosine_distance(
                FoodNutrition.embedding,
                func.cast(query_embedding, Vector)
            )
            
            # Simple similarity expressions without complex case statements
            name_similarity_expr = func.similarity(FoodNutrition.food_name, query)
            word_similarity_expr = func.word_similarity(query, FoodNutrition.food_name)
            
            # Text search rank - simplified
            if tsquery:
                ts_query_obj = func.plainto_tsquery('english', query)
                text_rank_expr = func.ts_rank_cd(FoodNutrition.document, ts_query_obj)
                has_text_match = FoodNutrition.document.op('@@')(ts_query_obj)
            else:
                text_rank_expr = func.literal(0.0)
                has_text_match = text('false')
            
            # Build WHERE conditions - simplified
            where_conditions = or_(
                semantic_distance_expr < 0.8,
                name_similarity_expr > fuzzy_threshold,
                word_similarity_expr > fuzzy_threshold,
                has_text_match
            )
            
            search_query = (
                select(
                    FoodNutrition,
                    semantic_distance_expr.label('semantic_distance'),
                    text_rank_expr.label('text_rank'),
                    name_similarity_expr.label('name_similarity'),
                    word_similarity_expr.label('word_similarity')
                )
                .where(where_conditions)
                .order_by(
                    # Simplified scoring
                    (semantic_distance_expr * 0.5 + 
                     (1.0 - text_rank_expr) * 0.2 + 
                     (1.0 - name_similarity_expr) * 0.15 +
                     (1.0 - word_similarity_expr) * 0.15).asc(),
                    FoodNutrition.id.asc()
                )
                .limit(limit * 2)  # Get more results to account for deduplication
            )
        else:
            # Fallback to text-only search
            name_similarity_expr = func.similarity(FoodNutrition.food_name, query)
            word_similarity_expr = func.word_similarity(query, FoodNutrition.food_name)
            
            if tsquery:
                ts_query_obj = func.plainto_tsquery('english', query)
                text_rank_expr = func.ts_rank_cd(FoodNutrition.document, ts_query_obj)
                has_text_match = FoodNutrition.document.op('@@')(ts_query_obj)
            else:
                text_rank_expr = func.literal(0.0)
                has_text_match = text('false')
            
            # Build WHERE conditions for text-only search
            where_conditions = or_(
                name_similarity_expr > fuzzy_threshold,
                word_similarity_expr > fuzzy_threshold,
                has_text_match
            )
            
            search_query = (
                select(
                    FoodNutrition,
                    func.literal(1.0).label('semantic_distance'),
                    text_rank_expr.label('text_rank'),
                    name_similarity_expr.label('name_similarity'),
                    word_similarity_expr.label('word_similarity')
                )
                .where(where_conditions)
                .order_by(
                    (text_rank_expr * 0.4 + 
                     name_similarity_expr * 0.3 + 
                     word_similarity_expr * 0.3).desc(),
                    FoodNutrition.id.asc()
                )
                .limit(limit * 2)
            )
        
        result = await session.execute(search_query)
        rows = result.all()
        
        # Additional deduplication based on food name and description similarity
        seen_foods = set()
        unique_results = []
        
        for row in rows:
            food_key = (row[0].food_name.lower().strip(), row[0].description.lower().strip())
            if food_key not in seen_foods:
                seen_foods.add(food_key)
                unique_results.append(row)
                if len(unique_results) >= limit:
                    break
        
        return unique_results

    def _format_search_results(
        self, 
        items: List[Tuple], 
        query: str,
        has_embedding: bool
    ) -> List[SearchResult]:
        """Format the search results efficiently with improved scoring"""
        results = []
        
        for row in items:
            item = row[0]  # FoodNutrition object
            semantic_distance = row[1] if len(row) > 1 else 1.0
            text_rank = row[2] if len(row) > 2 else 0.0
            name_similarity = row[3] if len(row) > 3 else 0.0
            word_similarity = row[4] if len(row) > 4 else 0.0
            
            # Calculate scores with better normalization
            semantic_score = max(0.0, 1.0 - float(semantic_distance)) if has_embedding else 0.0
            nlp_score = min(1.0, float(text_rank)) if text_rank else 0.0
            name_fuzzy_score = max(0.0, min(1.0, float(name_similarity))) if name_similarity else 0.0
            word_fuzzy_score = max(0.0, min(1.0, float(word_similarity))) if word_similarity else 0.0
            
            # Combine the fuzzy scores
            fuzzy_score = max(name_fuzzy_score, word_fuzzy_score)
            
            # Special boost for very close string matches (typo handling)
            typo_boost = 0.0
            if query and item.food_name:
                query_lower = query.lower()
                name_lower = item.food_name.lower()
                
                # Exact match
                if query_lower == name_lower:
                    typo_boost = 1.0
                # Contains or is contained
                elif query_lower in name_lower or name_lower in query_lower:
                    typo_boost = 0.8
                # Very similar (likely typo)
                elif len(query) > 2 and len(item.food_name) > 2:
                    import difflib
                    similarity_ratio = difflib.SequenceMatcher(None, query_lower, name_lower).ratio()
                    if similarity_ratio > 0.7:
                        typo_boost = similarity_ratio * 0.9
            
            # Combine scores with typo boost
            if has_embedding:
                combined_score = (
                    semantic_score * 0.35 + 
                    nlp_score * 0.15 + 
                    fuzzy_score * 0.25 + 
                    typo_boost * 0.25
                )
            else:
                combined_score = (
                    nlp_score * 0.3 + 
                    fuzzy_score * 0.4 + 
                    typo_boost * 0.3
                )
            
            results.append(
                SearchResult(
                    id=item.id,
                    food_name=item.food_name,
                    description=item.description,
                    nutrition=NutritionInfo(
                        energy_kcal=item.energy_kcal,
                        protein_g=item.protein_g,
                        fat_g=item.fat_g,
                        carbohydrates_g=item.carbohydrates_g,
                        fiber_g=item.fiber_g,
                        minerals={
                            "calcium_mg": item.calcium_mg,
                            "iron_mg": item.iron_mg,
                            "magnesium_mg": item.magnesium_mg,
                            "phosphorus_mg": item.phosphorus_mg,
                            "potassium_mg": item.potassium_mg,
                            "sodium_mg": item.sodium_mg,
                            "zinc_mg": item.zinc_mg
                        },
                        vitamins={
                            "vitamin_c_mg": item.vitamin_c_mg,
                            "thiamin_mg": item.thiamin_mg,
                            "riboflavin_mg": item.riboflavin_mg,
                            "niacin_mg": item.niacin_mg,
                            "vitamin_b6_mg": item.vitamin_b6_mg,
                            "folate_ug": item.folate_ug,
                            "vitamin_a_ug": item.vitamin_a_ug,
                            "vitamin_e_mg": item.vitamin_e_mg,
                            "vitamin_d_ug": item.vitamin_d_ug
                        }
                    ),
                    image_url=item.image_url,
                    scores={
                        "semantic": round(semantic_score, 4),
                        "nlp": round(nlp_score, 4),
                        "fuzzy": round(fuzzy_score, 4),
                        "typo_boost": round(typo_boost, 4),
                        "combined": round(combined_score, 4)
                    }
                )
            )
        
        # Sort by combined score (descending) for final ranking
        results.sort(key=lambda x: x.scores["combined"], reverse=True)
        
        return results

    async def search(self, query: str, limit: int = 10, use_cache: bool = True) -> Dict[str, Any]:
        """
        Perform hybrid search with optimized performance and error handling
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            use_cache: Whether to use caching
            
        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.time()
        cache_hit = False
        
        # Input validation
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if limit <= 0 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        # Clean and normalize the query
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty after cleaning")
        
        # Check cache first using the updated cache service
        cache_key = self._generate_cache_key(query, limit)
        if use_cache:
            try:
                cached_result = await cache_service.get_search_results(query, limit)
                if cached_result:
                    # Add cache hit indicator
                    cached_result["meta"]["cache"] = "HIT"
                    cached_result["meta"]["duration_ms"] = round((time.time() - start_time) * 1000, 2)
                    logger.info(f"Cache HIT for query: '{query}' (limit: {limit})")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache retrieval failed for query '{query}': {e}")
        
        async with self._session_factory() as session:
            try:
                # Get embedding with fallback and caching
                query_embedding = await self._get_embedding_with_fallback(query)
                has_embedding = query_embedding is not None
                
                # Execute search
                search_results = await self._execute_hybrid_search(
                    session, query, query_embedding, limit
                )
                
                # Format results
                results = self._format_search_results(search_results, query, has_embedding)
                
                # Calculate duration
                duration = (time.time() - start_time) * 1000
                
                # Prepare response
                response = {
                    "results": results,
                    "meta": {
                        "duration_ms": round(duration, 2),
                        "cache": "MISS",
                        "total_results": len(results),
                        "query": query,
                        "embedding_used": has_embedding,
                        "limit": limit
                    }
                }
                
                # Cache the result using the updated cache service
                if use_cache and results:
                    try:
                        await cache_service.set_search_results(query, limit, response)
                        logger.info(f"Cached search results for query: '{query}' (limit: {limit})")
                    except Exception as e:
                        logger.warning(f"Cache storage failed for query '{query}': {e}")
                
                logger.info(f"Search completed for query: '{query}' - {len(results)} results in {duration:.2f}ms")
                return response
                
            except SQLAlchemyError as e:
                logger.error(f"Database error in search: {str(e)}")
                await session.rollback()
                raise
            except Exception as e:
                logger.error(f"Unexpected error in search: {str(e)}")
                await session.rollback()
                raise

    async def search_by_nutrition(
        self, 
        nutrition_filters: Dict[str, Any], 
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search foods by nutritional criteria
        
        Args:
            nutrition_filters: Dict containing nutritional filters
            limit: Maximum number of results
            
        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.time()
        
        async with self._session_factory() as session:
            try:
                query = select(FoodNutrition)
                
                # Apply nutritional filters
                for key, value in nutrition_filters.items():
                    if hasattr(FoodNutrition, key) and value is not None:
                        if isinstance(value, dict):
                            # Range filter: {"min": 10, "max": 50}
                            if "min" in value:
                                query = query.where(getattr(FoodNutrition, key) >= value["min"])
                            if "max" in value:
                                query = query.where(getattr(FoodNutrition, key) <= value["max"])
                        else:
                            # Exact match or minimum value
                            query = query.where(getattr(FoodNutrition, key) >= value)
                
                query = query.limit(limit)
                result = await session.execute(query)
                items = result.scalars().all()
                
                # Format results (without scores since this is nutrition-based)
                results = []
                for item in items:
                    results.append(
                        SearchResult(
                            id=item.id,
                            food_name=item.food_name,
                            description=item.description,
                            nutrition=NutritionInfo(
                                energy_kcal=item.energy_kcal,
                                protein_g=item.protein_g,
                                fat_g=item.fat_g,
                                carbohydrates_g=item.carbohydrates_g,
                                fiber_g=item.fiber_g,
                                minerals={
                                    "calcium_mg": item.calcium_mg,
                                    "iron_mg": item.iron_mg,
                                    "magnesium_mg": item.magnesium_mg,
                                    "phosphorus_mg": item.phosphorus_mg,
                                    "potassium_mg": item.potassium_mg,
                                    "sodium_mg": item.sodium_mg,
                                    "zinc_mg": item.zinc_mg
                                },
                                vitamins={
                                    "vitamin_c_mg": item.vitamin_c_mg,
                                    "thiamin_mg": item.thiamin_mg,
                                    "riboflavin_mg": item.riboflavin_mg,
                                    "niacin_mg": item.niacin_mg,
                                    "vitamin_b6_mg": item.vitamin_b6_mg,
                                    "folate_ug": item.folate_ug,
                                    "vitamin_a_ug": item.vitamin_a_ug,
                                    "vitamin_e_mg": item.vitamin_e_mg,
                                    "vitamin_d_ug": item.vitamin_d_ug
                                }
                            ),
                            image_url=item.image_url,
                            scores={}
                        )
                    )
                
                duration = (time.time() - start_time) * 1000
                
                return {
                    "results": results,
                    "meta": {
                        "duration_ms": round(duration, 2),
                        "total_results": len(results),
                        "filters": nutrition_filters,
                        "limit": limit
                    }
                }
                
            except SQLAlchemyError as e:
                logger.error(f"Database error in nutrition search: {str(e)}")
                await session.rollback()
                raise
            except Exception as e:
                logger.error(f"Unexpected error in nutrition search: {str(e)}")
                await session.rollback()
                raise

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the search service"""
        try:
            async with self._session_factory() as session:
                # Simple query to test database connectivity
                result = await session.execute(select(func.count(FoodNutrition.id)))
                total_foods = result.scalar()
                
                return {
                    "status": "healthy",
                    "total_foods": total_foods,
                    "embedding_service": await embedding_service.health_check() if hasattr(embedding_service, 'health_check') else "unknown"
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Create a singleton instance
search_service = SearchService()

async def init_search_service():
    """Initialize the search service"""
    await search_service.initialize()
    logger.info("Search service initialization completed")
from fastapi import FastAPI, Request, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import logging
import asyncio
import gc
import weakref
from typing import Optional, Dict, Any
from functools import lru_cache
import orjson
from contextlib import asynccontextmanager

from .database import get_db, engine, async_session
from .auth import store_api_key, verify_client
from .search import search_service, init_search_service
from . import models, schemas
from .schemas import SearchParams
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Base
from sqlalchemy import text
from .init_data import data_initializer

# Optimized logging - single handler, buffer writes
logging.basicConfig(
    level=logging.WARNING,  # Changed to WARNING to reduce I/O
    format='%(name)s:%(levelname)s:%(message)s',  # Shorter format
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Optimized metrics with fewer buckets and labels
SEARCH_LATENCY = Histogram(
    'search_latency_seconds',
    'Search request latency',
    buckets=[0.1, 0.5, 1.0]  # Minimal buckets
)

SEARCH_REQUESTS = Counter(
    'search_requests_total',
    'Total search requests'
)

# Global state management
class AppState:
    __slots__ = ('startup_complete', 'db_initialized', 'search_initialized', 'cleanup_task')
    
    def __init__(self):
        self.startup_complete = False
        self.db_initialized = False
        self.search_initialized = False
        self.cleanup_task = None

app_state = AppState()

# Background cleanup task
async def cleanup_task():
    """Periodic cleanup to manage memory"""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        try:
            # Clear LRU caches
            verify_client_cached.cache_clear()
            # Force garbage collection
            collected = gc.collect()
            if collected > 100:  # Only log if significant cleanup
                logger.info(f"Cleaned up {collected} objects")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

@lru_cache(maxsize=256)  # Cache client verifications
async def verify_client_cached(client_id: str, db_session_id: int):
    """Cached client verification"""
    # Note: In production, implement proper cache invalidation
    async with async_session() as session:
        return await verify_client(session, client_id)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await initialize_database()
        await init_search_service()
        await initialize_embeddings()
        
        # Start cleanup task
        app_state.cleanup_task = asyncio.create_task(cleanup_task())
        app_state.startup_complete = True
        
        logger.info("Application started")
        yield
        
    finally:
        # Shutdown
        if app_state.cleanup_task:
            app_state.cleanup_task.cancel()
        
        await engine.dispose()
        gc.collect()
        logger.info("Application shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Food Search API",
    version="1.0.0",
    docs_url=None,  # Disabled for production
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan
)

# Minimal CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://nlp-search-two.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "x-client-id"],
    max_age=86400  # Cache for 24 hours
)

# Lightweight metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # Skip metrics for health/metrics endpoints
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        # Record metrics efficiently
        duration = time.perf_counter() - start_time
        SEARCH_LATENCY.observe(duration)
        SEARCH_REQUESTS.inc()

# Database initialization
async def initialize_database():
    """Optimized database initialization"""
    if app_state.db_initialized:
        return
    
    try:
        async with engine.begin() as conn:
            # Create extensions in parallel
            extensions = [
                "CREATE EXTENSION IF NOT EXISTS vector",
                "CREATE EXTENSION IF NOT EXISTS pg_trgm"
            ]
            
            for ext in extensions:
                await conn.execute(text(ext))
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
            # Create indexes - only essential ones
            essential_indexes = [
                """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_food_embedding 
                   ON food_nutrition USING ivfflat (embedding vector_cosine_ops) 
                   WITH (lists = 50)""",  # Reduced lists for faster queries
                
                """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_food_name_trgm 
                   ON food_nutrition USING GIN (food_name gin_trgm_ops)""",
                
                """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nutrition_range
                   ON food_nutrition (energy_kcal, protein_g) WHERE energy_kcal > 0"""
            ]
            
            for index_sql in essential_indexes:
                try:
                    await conn.execute(text(index_sql))
                except Exception as e:
                    logger.warning(f"Index creation skipped: {e}")
            
            app_state.db_initialized = True
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def initialize_embeddings():
    """Initialize embeddings with memory management"""
    try:
        async with async_session() as session:
            await data_initializer.initialize_embeddings(session)
        
        # Force cleanup after initialization
        gc.collect()
        
    except Exception as e:
        logger.error(f"Embeddings initialization failed: {e}")
        raise

# Optimized health check
@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "OK" if app_state.startup_complete else "STARTING"

# Metrics endpoint
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return generate_latest().decode('utf-8')

# Optimized key creation
@app.post("/api/keys")
async def create_key(
    client_id: str = Query(..., min_length=1, max_length=32),
    db: AsyncSession = Depends(get_db)
):
    try:
        # Simple deterministic key generation
        import hashlib
        timestamp = str(int(time.time()))
        key_data = f"{client_id}:{timestamp}".encode()
        key_hash = hashlib.sha256(key_data).hexdigest()[:16]
        api_key = f"DTwin_{key_hash}"
        
        await store_api_key(db, client_id, api_key)
        return {"api_key": api_key}
        
    except Exception as e:
        logger.error(f"Key creation error: {e}")
        raise HTTPException(status_code=500, detail="Key creation failed")

# Fast client verification
async def get_verified_client(request: Request, db: AsyncSession = Depends(get_db)) -> str:
    """Optimized client verification dependency"""
    client_id = request.headers.get("x-client-id")
    if not client_id:
        raise HTTPException(status_code=401, detail="Client ID required")
    
    try:
        # Use cached verification with session ID as cache key
        session_id = id(db)
        await verify_client_cached(client_id, session_id)
        return client_id
        
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid client")

# Main search endpoint - optimized
@app.post("/api/search")
async def search_endpoint(
    params: SearchParams,
    client_id: str = Depends(get_verified_client),
    db: AsyncSession = Depends(get_db)
):
    try:
        # Validate and sanitize input
        query = params.query.strip()
        if not query or len(query) > 200:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Invalid query")
        
        limit = min(max(params.limit, 1), 15)  # Cap at 15 for memory efficiency
        
        # Perform search with timeout
        try:
            results = await asyncio.wait_for(
                search_service.search(
                    query=query,
                    limit=limit,
                    use_cache=True
                ),
                timeout=5.0  # 5 second timeout
            )
            
            return results
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Search timeout")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

# Nutrition search endpoint - optimized
@app.post("/api/search/nutrition")
async def nutrition_search(
    filters: Dict[str, Any],
    limit: int = Query(10, ge=1, le=15),
    client_id: str = Depends(get_verified_client),
    db: AsyncSession = Depends(get_db)
):
    try:
        # Validate filters
        if not isinstance(filters, dict) or len(filters) > 10:
            raise HTTPException(status_code=400, detail="Invalid filters")
        
        # Sanitize numeric filters
        numeric_filters = {}
        for key, value in filters.items():
            if isinstance(value, (int, float)) and key.endswith(('_kcal', '_g', '_mg')):
                numeric_filters[key] = max(0, min(value, 10000))  # Reasonable bounds
        
        if not numeric_filters:
            raise HTTPException(status_code=400, detail="No valid filters")
        
        results = await asyncio.wait_for(
            search_service.search_by_nutrition(
                filters=numeric_filters,
                limit=limit
            ),
            timeout=3.0
        )
        
        return results
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Search timeout")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Nutrition search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

# Cache stats endpoint (for monitoring)
@app.get("/api/stats")
async def cache_stats(client_id: str = Depends(get_verified_client)):
    return {
        "client_cache": verify_client_cached.cache_info()._asdict(),
        "memory_objects": len(gc.get_objects()),
        "search_service_ready": app_state.search_initialized
    }

# Production server configuration
def create_production_app():
    """Factory for production deployment"""
    return app

# Development runner
if __name__ == "__main__":
    import uvicorn
    
    # Optimized uvicorn config for development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False,
        workers=1,
        loop="asyncio",
        # Performance optimizations
        limit_concurrency=50,
        limit_max_requests=500,
        timeout_keep_alive=3,
        backlog=512,
        # Memory optimizations
        h11_max_incomplete_event_size=8192,
        ws_max_size=1024*1024,  # 1MB WebSocket limit
        # Logging
        log_level="warning",
        use_colors=False
    )
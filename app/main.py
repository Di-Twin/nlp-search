from fastapi import FastAPI, Request, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
import time
import json
from .database import get_db, engine, async_session
from .auth import store_api_key, verify_client, ClientKey
from .search import search_service, init_search_service
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from . import models, schemas, database
from .schemas import SearchParams
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Base
import logging
import asyncio
from .metrics import metrics_middleware
from sqlalchemy import text
from .init_data import data_initializer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Food Search API",
    description="High-performance NLP food search service with hybrid retrieval",
    version="1.0.0"
)

# Custom metrics
SEARCH_LATENCY = Histogram(
    'food_search_latency_seconds',
    'Time spent processing search requests',
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
)

SEARCH_REQUESTS = Counter(
    'food_search_requests_total',
    'Total number of search requests',
    ['status']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://nlp-search-two.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.middleware("http")(metrics_middleware)

# Initialize database on startup
@app.on_event("startup")
async def startup():
    try:
        # Create tables
        async with engine.begin() as conn:
            # Create extensions first
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            
            # Create tables using async DDL
            await conn.run_sync(Base.metadata.create_all)
            
            # Create indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS food_nutrition_embedding_idx 
                ON food_nutrition 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS food_nutrition_document_idx 
                ON food_nutrition 
                USING GIN (document)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS food_nutrition_food_name_trgm_idx 
                ON food_nutrition 
                USING GIN (food_name gin_trgm_ops)
            """))
        
        # Initialize search service
        await init_search_service()
        
        # Initialize embeddings for existing data
        async with async_session() as session:
            await data_initializer.initialize_embeddings(session)
            
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

# Health check endpoint (no auth required)
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Key generation endpoint (no auth required)
@app.post("/api/keys")
async def create_api_key(
    client_id: str = Query(..., description="Client ID for the API key"),
    db: AsyncSession = Depends(get_db)
):
    try:
        # Generate a new API key
        api_key = "DTwin"  # In production, generate a secure random key
        
        # Store the API key in the database
        await store_api_key(db, client_id, api_key)
        
        return {"api_key": api_key}
    except Exception as e:
        logger.error(f"Error creating API key: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create API key")

# Search endpoint (requires client ID)
@app.post("/api/search")
async def search(
    request: Request,
    search_params: SearchParams,
    db: AsyncSession = Depends(get_db)
):
    """Search endpoint with client verification"""
    try:
        # Verify client
        client_id = request.headers.get("X-Client-Id")
        if not client_id:
            raise HTTPException(status_code=401, detail="Client ID required")
        
        # Verify client
        await verify_client(db, client_id)

        # Start timing
        start_time = time.time()
        
        # Perform search using validated parameters
        results = await search_service.search(
            query=search_params.query,
            limit=search_params.limit,
            use_cache=True
        )
        
        # Record metrics
        latency = time.time() - start_time
        SEARCH_LATENCY.observe(latency)
        SEARCH_REQUESTS.labels(status="success").inc()
        
        return results
    except ValueError as e:
        # Record error metrics
        SEARCH_REQUESTS.labels(status="error").inc()
        logger.error(f"Invalid search parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Record error metrics
        SEARCH_REQUESTS.labels(status="error").inc()
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    ) 
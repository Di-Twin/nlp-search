from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text
from sqlalchemy.pool import NullPool
import os
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database URL from environment variable and remove sslmode parameter
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@db:5432/food_nutrition")
if "sslmode=" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split("?")[0]  # Remove query parameters

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    poolclass=NullPool,  # Disable connection pooling for async
    connect_args={
        "server_settings": {
            "application_name": "food_nlp_search"
        }
    }
)

# Create async session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Create base class for models
Base = declarative_base()

# Dependency to get database session
async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Initialize database
async def init_db():
    try:
        async with engine.begin() as conn:
            # Create extensions
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
                USING GIN (to_tsvector('english', document))
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS food_nutrition_food_name_trgm_idx 
                ON food_nutrition 
                USING GIN (food_name gin_trgm_ops)
            """))
            
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def get_search_service():
    """Get search service instance"""
    from .search import search_service
    return search_service 
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func
from .models import FoodNutrition
from .embeddings import embedding_service
import logging
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class DataInitializer:
    def __init__(self, batch_size: int = 1000, max_workers: int = 8):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def initialize_data(self, db: AsyncSession):
        """Initialize embeddings for existing food data"""
        try:
            # Get all food items without embeddings
            result = await db.execute(
                select(FoodNutrition).filter(FoodNutrition.embedding.is_(None))
            )
            food_items = result.scalars().all()
            
            if not food_items:
                logger.info("No new items to initialize embeddings for")
                return

            logger.info(f"Initializing embeddings for {len(food_items)} items")
            
            # Process items in batches
            for i in range(0, len(food_items), self.batch_size):
                batch = food_items[i:i + self.batch_size]
                await self._process_batch(db, batch)

            logger.info("Embedding initialization completed successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise

    async def _process_batch(self, db: AsyncSession, batch: List[FoodNutrition]):
        """Process a batch of food items"""
        try:
            # Generate embeddings for the batch
            texts = [f"{item.food_name}. {item.description}" for item in batch]
            embeddings = await self._generate_embeddings_batch(texts)

            # Update items with embeddings
            for item, embedding in zip(batch, embeddings):
                item.embedding = embedding

            # Commit changes
            await db.commit()
            logger.info(f"Processed batch of {len(batch)} items")
        except Exception as e:
            await db.rollback()
            logger.error(f"Error processing batch: {str(e)}")
            raise

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using thread pool"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                embedding_service.generate_embeddings_batch,
                texts
            )
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    async def initialize_embeddings(self, db: AsyncSession):
        """Initialize embeddings for all food items in the database"""
        start_time = time.time()
        logger.info("Starting embedding initialization...")

        try:
            # Get total count of items
            result = await db.execute(select(func.count()).select_from(FoodNutrition))
            total_count = result.scalar()
            logger.info(f"Found {total_count} items to process")

            # Process in batches
            processed = 0
            while processed < total_count:
                # Get batch of items without embeddings
                result = await db.execute(
                    select(FoodNutrition)
                    .filter(FoodNutrition.embedding.is_(None))
                    .limit(self.batch_size)
                )
                items = result.scalars().all()

                if not items:
                    break

                # Process batch
                await self._process_batch(db, items)
                processed += len(items)
                logger.info(f"Processed {processed}/{total_count} items")

            duration = time.time() - start_time
            logger.info(f"Embedding initialization completed in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during embedding initialization: {str(e)}")
            raise

# Global data initializer instance
data_initializer = DataInitializer() 
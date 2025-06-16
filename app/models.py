from sqlalchemy import Column, Integer, String, Float, Text, Index, text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector
from .database import Base

class Base(AsyncAttrs, DeclarativeBase):
    pass

class FoodNutrition(Base):
    __tablename__ = "food_nutrition"

    id = Column(Integer, primary_key=True, index=True)
    food_name = Column(String, index=True)
    description = Column(Text)
    energy_kcal = Column(Float)
    protein_g = Column(Float)
    fat_g = Column(Float)
    carbohydrates_g = Column(Float)
    fiber_g = Column(Float)
    calcium_mg = Column(Float)
    iron_mg = Column(Float)
    magnesium_mg = Column(Float)
    phosphorus_mg = Column(Float)
    potassium_mg = Column(Float)
    sodium_mg = Column(Float)
    zinc_mg = Column(Float)
    vitamin_c_mg = Column(Float)
    thiamin_mg = Column(Float)
    riboflavin_mg = Column(Float)
    niacin_mg = Column(Float)
    vitamin_b6_mg = Column(Float)
    folate_ug = Column(Float)
    vitamin_a_ug = Column(Float)
    vitamin_e_mg = Column(Float)
    vitamin_d_ug = Column(Float)
    image_url = Column(String)
    embedding = Column(Vector(384))  # Dimension for all-MiniLM-L6-v2 model
    document = Column(
        TSVECTOR,
        server_default=text("to_tsvector('english', coalesce(food_name,'') || ' ' || coalesce(description,''))")
    )

    # Create indexes
    __table_args__ = (
        Index('food_nutrition_embedding_idx', embedding, postgresql_using='ivfflat', postgresql_with={'lists': 100}),
        Index('food_nutrition_document_idx', document, postgresql_using='gin'),
        Index('food_nutrition_food_name_trgm_idx', food_name, postgresql_using='gin', postgresql_ops={'food_name': 'gin_trgm_ops'})
    ) 
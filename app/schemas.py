from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

class SearchParams(BaseModel):
    query: str = Field(..., description="Search query string")
    limit: int = Field(default=8, ge=1, le=100, description="Maximum number of results to return")

    @validator('query')
    def validate_query(cls, v):
        """Ensure query is a non-empty string"""
        if not v or not isinstance(v, str):
            raise ValueError("Query must be a non-empty string")
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty after cleaning")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "low carb high protein dinner",
                "limit": 8
            }
        }

class NutritionInfo(BaseModel):
    energy_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    fiber_g: Optional[float] = None
    minerals: Optional[Dict[str, float]] = None
    vitamins: Optional[Dict[str, float]] = None

class SearchResult(BaseModel):
    id: int
    food_name: str
    description: Optional[str] = None
    nutrition: Optional[NutritionInfo] = None
    image_url: Optional[str] = None
    scores: Optional[Dict[str, float]] = None

    class Config:
        from_attributes = True

class SearchMeta(BaseModel):
    duration_ms: float
    cache: str
    total_results: int
    query: str
    embedding_used: bool
    limit: int

class SearchResponse(BaseModel):
    results: List[SearchResult]
    meta: SearchMeta

    class Config:
        from_attributes = True

class ApiKeyResponse(BaseModel):
    api_key: str

class FoodItem(BaseModel):
    id: int
    food_name: str
    description: Optional[str] = None
    calories: Optional[float] = None
    protein: Optional[float] = None
    fat: Optional[float] = None
    carbohydrates: Optional[float] = None
    fiber: Optional[float] = None
    sugar: Optional[float] = None
    sodium: Optional[float] = None
    scores: Optional[Dict[str, float]] = None

    class Config:
        from_attributes = True 
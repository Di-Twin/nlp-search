import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db
from app.search import search_service
import os

# Test database URL
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/food_search_test"

# Create test engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def test_db():
    # Create test database tables
    Base.metadata.create_all(bind=engine)
    
    # Create test session
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Drop test database tables
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            test_db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

def test_search_endpoint(client):
    # Test data
    test_food = {
        "food_name": "Grilled Chicken Salad",
        "description": "Healthy salad with grilled chicken breast, mixed greens, and vinaigrette",
        "energy_kcal": 350,
        "protein_g": 30,
        "carbs_g": 15,
        "fat_g": 18,
        "image_url": "https://example.com/chicken-salad.jpg"
    }
    
    # Insert test data
    response = client.post(
        "/api/foods",
        json=test_food,
        headers={
            "X-Client-Id": "test_client"
        }
    )
    assert response.status_code == 200
    
    # Test search
    search_query = {
        "query": "healthy chicken salad",
        "limit": 5
    }
    
    response = client.post(
        "/api/search",
        json=search_query,
        headers={
            "X-Client-Id": "test_client"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "results" in data
    assert "meta" in data
    assert "duration_ms" in data["meta"]
    assert "cache" in data["meta"]
    assert "total_results" in data["meta"]
    assert "query" in data["meta"]
    assert "embedding_used" in data["meta"]
    assert "limit" in data["meta"]
    
    # Verify search results
    assert len(data["results"]) > 0
    first_result = data["results"][0]
    assert "id" in first_result
    assert "food_name" in first_result
    assert "description" in first_result
    assert "nutrition" in first_result
    assert "scores" in first_result
    
    # Verify scores
    scores = first_result["scores"]
    assert "semantic" in scores
    assert "nlp" in scores
    assert "fuzzy" in scores
    assert "typo_boost" in scores
    assert "combined" in scores
    assert all(0 <= score <= 1 for score in scores.values())

def test_search_performance(client):
    """Test that search response time is within acceptable limits"""
    search_query = {
        "query": "healthy dinner options",
        "limit": 8
    }
    
    response = client.post(
        "/api/search",
        json=search_query,
        headers={
            "X-Client-Id": "test_client"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response time is under 20ms
    assert data["meta"]["duration_ms"] < 20

def test_search_cache(client):
    """Test that search results are properly cached"""
    search_query = {
        "query": "protein rich foods",
        "limit": 5
    }
    
    # First request (cache miss)
    response1 = client.post(
        "/api/search",
        json=search_query,
        headers={
            "X-Client-Id": "test_client"
        }
    )
    assert response1.status_code == 200
    assert response1.json()["meta"]["cache"] == "MISS"
    
    # Second request (should be cache hit)
    response2 = client.post(
        "/api/search",
        json=search_query,
        headers={
            "X-Client-Id": "test_client"
        }
    )
    assert response2.status_code == 200
    assert response2.json()["meta"]["cache"] == "HIT"
    
    # Verify results are identical
    assert response1.json()["results"] == response2.json()["results"]

def test_search_validation(client):
    """Test search parameter validation"""
    # Test empty query
    response = client.post(
        "/api/search",
        json={"query": "", "limit": 5},
        headers={"X-Client-Id": "test_client"}
    )
    assert response.status_code == 400
    
    # Test invalid limit
    response = client.post(
        "/api/search",
        json={"query": "test", "limit": 0},
        headers={"X-Client-Id": "test_client"}
    )
    assert response.status_code == 400
    
    # Test missing client ID
    response = client.post(
        "/api/search",
        json={"query": "test", "limit": 5}
    )
    assert response.status_code == 401 
# High-Performance NLP Food Search Service

A production-grade FastAPI service that provides ultra-fast food search capabilities using a hybrid approach combining semantic embeddings, traditional NLP, and fuzzy matching.

## Features

- 🚀 Sub-20ms P95 response times
- 🔍 Hybrid search combining:
  - Semantic (vector) search via pgvector
  - NLP (full-text) search via PostgreSQL
  - Fuzzy matching via trigram similarity
- 🔐 Public/Private key authentication
- 💾 Redis caching for queries and embeddings
- 📊 Prometheus metrics and monitoring
- 🐳 Docker and docker-compose support

## Prerequisites

- Docker and docker-compose
- Python 3.11+ (for local development)
- PostgreSQL 14+ with pgvector extension
- Redis 7+

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd food-nlp-search
```

2. Start the services:
```bash
docker-compose up -d
```

3. The API will be available at `http://localhost:8000`

## API Usage

### Authentication

The API uses public/private key authentication. Generate a key pair:

```bash
# Generate private key
openssl genpkey -algorithm RSA -out private.pem -pkeyopt rsa_keygen_bits:2048

# Extract public key
openssl rsa -pubout -in private.pem -out public.pem
```

### Making Requests

```bash
# Example search request
RAW='{"query":"low carb high protein dinner","limit":8}'
SIGN=$(echo -n "$RAW" | openssl dgst -sha256 -sign private.pem | base64)

curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -H "X-Client-Id: your_client_id" \
  -H "X-Signature: $SIGN" \
  -d "$RAW"
```

## Development

### Local Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the development server:
```bash
uvicorn app.main:app --reload
```

### Database Setup

The database schema is automatically created when the application starts. For manual setup:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS food_nutrition (
  id          SERIAL PRIMARY KEY,
  food_name   TEXT      NOT NULL,
  description TEXT      NOT NULL,
  energy_kcal FLOAT,
  protein_g   FLOAT,
  carbs_g     FLOAT,
  fat_g       FLOAT,
  image_url   TEXT,
  embedding   VECTOR(768),
  document    tsvector GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(food_name,'') || ' ' || coalesce(description,''))
  ) STORED
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_food_embedding 
  ON food_nutrition USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 512, probes = 10);

CREATE INDEX IF NOT EXISTS idx_food_document 
  ON food_nutrition USING gin (document);

CREATE INDEX IF NOT EXISTS idx_food_name_trgm 
  ON food_nutrition USING gin (food_name gin_trgm_ops);
```

## Performance Tuning

- Adjust the number of workers in `docker-compose.yml` based on your CPU cores
- Tune PostgreSQL parameters in `postgresql.conf`
- Configure Redis memory limits and eviction policies
- Monitor performance metrics at `/metrics` endpoint

## Monitoring

The service exposes Prometheus metrics at `/metrics`. Key metrics include:

- `http_request_duration_seconds`: Request latency
- `cache_hits_total`: Cache hit rate
- `http_requests_total`: Request count by endpoint

## License

MIT License 
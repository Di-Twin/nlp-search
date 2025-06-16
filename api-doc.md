# Food Search API Documentation

## Overview

The Food Search API provides high-performance semantic search capabilities for food items using a hybrid approach combining vector embeddings, full-text search, and fuzzy matching. The API is designed for sub-20ms response times and high throughput.

## Base URL

```
http://localhost:8000
```

## Authentication

The API uses a simple client ID authentication system. Each client needs to register and receive a client ID, which is used to identify and authenticate requests.

### Key Generation

To get started, generate an API key using the `/api/keys` endpoint:

```bash
# Generate a new API key
curl -X POST "http://localhost:8000/api/keys?client_id=your_client_id"
```

Response:
```json
{
  "api_key": "DTwin"
}
```

The API key is automatically stored on the server, and you only need to include your client ID in requests.

### Making Requests

Simply include your client ID in the `X-Client-Id` header:

```bash
curl -X POST https://api.example.com/api/search \
  -H "Content-Type: application/json" \
  -H "X-Client-Id: your_client_id" \
  -d '{
    "query": "low carb high protein dinner",
    "limit": 8
  }'
```

### Headers

| Header | Description |
|--------|-------------|
| `X-Client-Id` | Your client identifier |

## Endpoints

### Generate API Key

Generate a new API key for authentication.

```http
POST /api/keys
```

#### Query Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `client_id` | Yes | Your chosen client identifier |

#### Response

```json
{
  "api_key": "string"  // Your API key
}
```

### Search Food Items

Search for food items using natural language queries.

```http
POST /api/search
```

#### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-Client-Id` | Yes | Your client identifier |

#### Request Body

```json
{
  "query": "string",     // Search query
  "limit": integer      // Maximum number of results (1-100, default: 8)
}
```

#### Response

```json
{
  "results": [
    {
      "id": integer,
      "food_name": "string",
      "description": "string",
      "nutrition": {
        "energy_kcal": number,
        "protein_g": number,
        "fat_g": number,
        "carbohydrates_g": number,
        "fiber_g": number,
        "minerals": {
          "calcium_mg": number,
          "iron_mg": number,
          "magnesium_mg": number,
          "phosphorus_mg": number,
          "potassium_mg": number,
          "sodium_mg": number,
          "zinc_mg": number
        },
        "vitamins": {
          "vitamin_c_mg": number,
          "thiamin_mg": number,
          "riboflavin_mg": number,
          "niacin_mg": number,
          "vitamin_b6_mg": number,
          "folate_ug": number,
          "vitamin_a_ug": number,
          "vitamin_e_mg": number,
          "vitamin_d_ug": number
        }
      },
      "image_url": "string",
      "scores": {
        "semantic": number,    // Vector similarity score (0-1)
        "nlp": number,        // Full-text search score (0-1)
        "fuzzy": number,      // Fuzzy matching score (0-1)
        "typo_boost": number, // Typo handling boost (0-1)
        "combined": number    // Weighted combination of all scores (0-1)
      }
    }
  ],
  "meta": {
    "duration_ms": number,    // Request processing time in milliseconds
    "cache": "HIT|MISS",     // Cache status
    "total_results": number,  // Total number of results found
    "query": "string",       // Original search query
    "embedding_used": boolean, // Whether semantic search was used
    "limit": number          // Maximum results requested
  }
}
```

### Search by Nutrition

Search for foods based on nutritional criteria.

```http
POST /api/search/nutrition
```

#### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-Client-Id` | Yes | Your client identifier |

#### Request Body

```json
{
  "filters": {
    "energy_kcal": { "min": 100, "max": 500 },
    "protein_g": { "min": 20 },
    "fat_g": { "max": 30 }
  },
  "limit": integer  // Maximum number of results (1-100, default: 10)
}
```

#### Response

```json
{
  "results": [
    {
      "id": integer,
      "food_name": "string",
      "description": "string",
      "nutrition": {
        "energy_kcal": number,
        "protein_g": number,
        "fat_g": number,
        "carbohydrates_g": number,
        "fiber_g": number,
        "minerals": {
          "calcium_mg": number,
          "iron_mg": number,
          "magnesium_mg": number,
          "phosphorus_mg": number,
          "potassium_mg": number,
          "sodium_mg": number,
          "zinc_mg": number
        },
        "vitamins": {
          "vitamin_c_mg": number,
          "thiamin_mg": number,
          "riboflavin_mg": number,
          "niacin_mg": number,
          "vitamin_b6_mg": number,
          "folate_ug": number,
          "vitamin_a_ug": number,
          "vitamin_e_mg": number,
          "vitamin_d_ug": number
        }
      },
      "image_url": "string"
    }
  ],
  "meta": {
    "duration_ms": number,    // Request processing time in milliseconds
    "total_results": number,  // Total number of results found
    "filters": object,       // Applied nutritional filters
    "limit": number         // Maximum results requested
  }
}
```

### Health Check

Check the API health status.

```http
GET /health
```

#### Response

```json
{
  "status": "healthy"
}
```

## Error Responses

The API uses standard HTTP status codes and returns error details in the response body.

### Error Response Format

```json
{
  "detail": "Error message"
}
```

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid request format |
| 401 | Unauthorized - Missing client ID |
| 403 | Forbidden - Invalid client ID |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

## Rate Limiting

- 1000 requests per minute per client
- Rate limit headers included in responses:
  - `X-RateLimit-Limit`: Maximum requests per window
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Time until rate limit resets (Unix timestamp)

## Performance

- P95 latency: < 20ms
- Cache hit ratio: > 90% under normal load
- Maximum concurrent users: 10,000+
- Queries per day: Millions

## Best Practices

1. **Key Management**
   - Store your client ID securely
   - Use different client IDs for different environments
   - Monitor API usage per client ID

2. **Caching**
   - Cache responses on the client side
   - Respect cache headers
   - Implement stale-while-revalidate pattern

3. **Error Handling**
   - Implement exponential backoff for retries
   - Handle rate limit errors gracefully
   - Log failed requests for debugging

4. **Security**
   - Keep your client ID private
   - Use HTTPS for all requests
   - Monitor for suspicious activity

5. **Performance**
   - Keep request payloads small
   - Use appropriate limit values
   - Monitor response times

## Monitoring

The API exposes Prometheus metrics at `/metrics` for monitoring:

- Request latency
- Cache hit ratio
- Error rates
- Request volume

## Support

For API support, contact:
- Email: api-support@example.com
- Documentation: https://docs.example.com
- Status Page: https://status.example.com 
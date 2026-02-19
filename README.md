# 🚀 E-Commerce Product Classifier API

FastAPI server for XLM-RoBERTa product classification with Docker support.

## Features

- ✅ **Fast inference**: 50-200 samples/second
- ✅ **Top-K predictions**: Returns top 5 categories by default
- ✅ **Batch processing**: Classify multiple products efficiently
- ✅ **Docker ready**: Containerized for easy deployment
- ✅ **Model on-demand**: Downloads model on first startup (lighter image)
- ✅ **Health checks**: Built-in monitoring
- ✅ **Auto-documentation**: Interactive API docs at `/docs`

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start
docker-compose up --build

# Access API
curl http://localhost:8000/health
```

### Option 2: Docker

```bash
# Build image
docker build -t ecommerce-classifier-api .

# Run container
docker run -d \
  --name product-cls-service \
  -p 8000:8000 \
  -e HF_HOME=/app/.cache/huggingface \
  -v hf_cache:/app/.cache/huggingface \
  --restart unless-stopped \
  product-cls-service
```

### Option 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### 1. Classify Single Product

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sony WH-1000XM5 wireless headphones",
    "top_k": 5
  }'
```

**Response:**

```json
{
  "text": "Sony WH-1000XM5 wireless headphones",
  "predictions": [
    {
      "category_id": 0,
      "category_name": "electronics",
      "display_name": "Electronics",
      "score": 0.9534
    },
    {
      "category_id": 1,
      "category_name": "computers_networking",
      "display_name": "Computers & Networking",
      "score": 0.0234
    },
    {
      "category_id": 2,
      "category_name": "mobile_phones_tablets",
      "display_name": "Mobile Phones & Tablets",
      "score": 0.0123
    }
  ],
  "inference_time": 0.045
}
```

### 2. Batch Classification

```bash
curl -X POST "http://localhost:8000/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "iPhone 15 Pro Max",
      "Nike running shoes",
      "Samsung Smart TV"
    ],
    "top_k": 3
  }'
```

### 3. Get All Categories

```bash
curl http://localhost:8000/categories
```

**Response:**

```json
[
  {
    "category_id": 0,
    "name": "electronics",
    "display_name": "Electronics",
    "description": "Televisions, cameras, audio speakers, headphones..."
  },
  ...
]
```

### 4. Get Category by ID

```bash
curl http://localhost:8000/categories/0
```

### 5. Get Category by Name

```bash
curl http://localhost:8000/categories/name/electronics
```

## 📋 Category Reference

| ID  | Name                       | Display Name                 |
| --- | -------------------------- | ---------------------------- |
| 0   | electronics                | Electronics                  |
| 1   | computers_networking       | Computers & Networking       |
| 2   | mobile_phones_tablets      | Mobile Phones & Tablets      |
| 3   | fashion_clothing           | Fashion & Clothing           |
| 4   | shoes_footwear             | Shoes & Footwear             |
| 5   | bags_luggage               | Bags & Luggage               |
| 6   | watches                    | Watches                      |
| 7   | jewelry                    | Jewelry                      |
| 8   | fashion_accessories        | Fashion Accessories          |
| 9   | beauty_personal_care       | Beauty & Personal Care       |
| 10  | health_wellness            | Health & Wellness            |
| 11  | sports_outdoors            | Sports & Outdoors            |
| 12  | home_furniture             | Home Furniture               |
| 13  | home_decor_lighting        | Home Decor & Lighting        |
| 14  | kitchen_dining             | Kitchen & Dining             |
| 15  | bedding_bath               | Bedding & Bath               |
| 16  | large_appliances           | Large Appliances             |
| 17  | small_appliances           | Small Appliances             |
| 18  | grocery_food               | Grocery & Food               |
| 19  | baby_maternity             | Baby & Maternity             |
| 20  | toys_games                 | Toys & Games                 |
| 21  | books_media                | Books & Media                |
| 22  | stationery_office_supplies | Stationery & Office Supplies |
| 23  | pet_supplies               | Pet Supplies                 |
| 24  | automotive_motorcycle      | Automotive & Motorcycle      |
| 25  | tools_hardware             | Tools & Hardware             |
| 26  | garden_outdoor_living      | Garden & Outdoor Living      |
| 27  | musical_instruments        | Musical Instruments          |
| 28  | video_games_gaming         | Video Games & Gaming         |
| 29  | software_digital_goods     | Software & Digital Goods     |
| 30  | arts_crafts                | Arts & Crafts                |
| 31  | industrial_commercial      | Industrial & Commercial      |

## Interactive Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Response Format

### Single Classification

```python
{
  "text": str,                    # Input text
  "predictions": [               # Top-K predictions
    {
      "label": str,              # Category name
      "score": float             # Confidence (0-1)
    }
  ],
  "inference_time": float        # Time in seconds
}
```

### Batch Classification

```python
{
  "results": [                   # Results for each input
    {
      "text": str,
      "predictions": [...]
    }
  ],
  "total_time": float,           # Total processing time
  "count": int                   # Number of inputs
}
```

## Model Information

- **Model**: `Lezh1n/xlm-roberta-ecommerce-classifier`
- **Base**: XLM-RoBERTa-base
- **Categories**: 32
- **Accuracy**: 90.14%
- **F1 Score**: 90.00%

## Configuration

### Environment Variables

```bash
MODEL_NAME=Lezh1n/xlm-roberta-ecommerce-classifier
PORT=8000
WORKERS=1  # Number of uvicorn workers
```

### Docker Volumes

Model cache is persisted in a Docker volume to avoid re-downloading:

```yaml
volumes:
  - model-cache:/root/.cache/huggingface
```

## Performance

### Benchmarks (on CPU)

| Operation        | Time   | Throughput       |
| ---------------- | ------ | ---------------- |
| Single inference | ~45ms  | 22 req/sec       |
| Batch (10 items) | ~180ms | 55 items/sec     |
| Cold start       | ~15s   | (model download) |

### With GPU

| Operation         | Time   | Throughput    |
| ----------------- | ------ | ------------- |
| Single inference  | ~15ms  | 66 req/sec    |
| Batch (100 items) | ~800ms | 125 items/sec |

## Production Deployment

### With GPU Support

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ... rest of Dockerfile
```

```bash
docker run --gpus all -p 8000:8000 ecommerce-classifier-api
```

### Scaling with Docker Compose

```yaml
services:
  api:
    build: .
    deploy:
      replicas: 3 # Run 3 instances
    ports:
      - "8000-8002:8000"
```

### Behind Nginx

```nginx
upstream api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://api;
    }
}
```

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

Returns:

```json
{
  "status": "healthy",
  "model": "Lezh1n/xlm-roberta-ecommerce-classifier",
  "device": "cuda"
}
```

### Docker Health Check

Built-in health check runs every 30 seconds:

```bash
docker ps  # Check health status
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
pytest tests/
```

### Example Test

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_classify():
    response = client.post("/classify", json={
        "text": "iPhone 15 Pro",
        "top_k": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["label"] == "mobile_phones_tablets"
```

## Troubleshooting

### Model Download Issues

If model fails to download:

```bash
# Pre-download model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Lezh1n/xlm-roberta-ecommerce-classifier')"
```

### Out of Memory

Reduce batch size or use CPU:

```python
# In main.py
DEVICE = "cpu"  # Force CPU
```

### Slow Cold Start

First request downloads model (~1.1GB). Subsequent requests are fast.

To pre-download model in Docker:

```dockerfile
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('Lezh1n/xlm-roberta-ecommerce-classifier'); \
    AutoTokenizer.from_pretrained('Lezh1n/xlm-roberta-ecommerce-classifier')"
```

⚠️ **Note**: This increases image size from ~2GB to ~3GB.

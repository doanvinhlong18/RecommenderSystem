from web.backend.api_server import app
from fastapi.testclient import TestClient
import json

client = TestClient(app)
res = client.get('/api/compare?anime_id=20&top_k=5')
print(json.dumps(res.json(), indent=2))

# 03 RAG Gist Implementation Using Langchain

## Env Setup

### Pinecone Local Setup

```sh
docker pull ghcr.io/pinecone-io/pinecone-local:latest
```

```sh
docker run -d \
 --name pinecone-local \
 -e PORT=5080 \
 -e PINECONE_HOST=localhost \
 -p 5080-5090:5080-5090 \
 --platform linux/amd64 \
 ghcr.io/pinecone-io/pinecone-local:latest
```

From Python

```python
from pinecone import Pinecone

pc = Pinecone(
    api_key="pclocal",
    host="http://localhost:5080"
)
```

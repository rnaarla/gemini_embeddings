# Gemini Embedding (`gemini-embedding-001`) — README

## 📌 Overview

Gemini Embedding is a state-of-the-art text embedding model released by Google under the Gemini API. It converts text into high-dimensional numerical vectors optimized for semantic understanding. This README serves as a complete guide to using `gemini-embedding-001` for tasks such as RAG, semantic search, classification, and clustering.

## 🎯 Use Cases

| Application                | Benefit                                                  |
| -------------------------- | -------------------------------------------------------- |
| 🔍 Semantic Search         | Retrieve relevant text beyond keyword matching           |
| 🧠 Retrieval-Augmented Gen | Improve grounding of LLM outputs using vector similarity |
| 🗂️ Clustering             | Automatically group similar documents/emails             |
| 🏷️ Classification         | Label documents based on embedding similarity            |
| 🤝 Semantic Matching       | Match resumes to jobs, FAQs to queries, etc.             |

## 🔑 Key Features

- 🥇 #1 on MTEB multilingual leaderboard
- 🌍 Supports **100+ languages**
- 📏 Input size: Up to **2,048 tokens**
- 🧩 **Matryoshka Representation Learning**:
  - Flexible output sizes: 768, 1536, or 3072 dimensions
  - Balance between embedding cost and accuracy
- 🏛️ Domain-agnostic: Optimized across science, legal, finance, code

## ⚙️ How to Use

### Step 1: Get Access

- Use [Google AI Studio](https://makersuite.google.com/app) to explore Gemini models interactively.
- Or access via [Vertex AI](https://cloud.google.com/vertex-ai) on GCP for enterprise-grade deployment.

### Step 2: Prepare Input

- Input text can be a sentence, paragraph, or document.
- Total length must be within 2,048 tokens.

### Step 3: Call `embedContent`

- Use the Gemini API to call the `embedContent` method.
- You’ll receive a high-dimensional vector representing the input's semantic meaning.

### Step 4: Choose Embedding Dimensionality

- Specify output vector size (768, 1536, 3072) depending on:
  - Precision needed
  - Storage/computation cost constraints

### Step 5: Integrate with Vector Database

- Store embeddings in a vector DB like Pinecone, Weaviate, Qdrant, or FAISS
- Use cosine similarity or dot product for retrieval

## 🔗 Vector Database Integration Guide

### ✅ Gemini + Pinecone

This section demonstrates a production-grade integration pipeline using `gemini-embedding-001` and Pinecone for scalable vector storage and retrieval. It includes batch embedding, structured metadata handling, robust querying, and error management.

#### 🔹 Step 1: Generate Embeddings with Gemini API (Batch Mode)

```python
import google.generativeai as genai

genai.configure(api_key='YOUR_GEMINI_API_KEY')
model = genai.GenerativeModel('gemini-embedding-001')

def get_embeddings(texts, task_type="retrieval_document"):
    return [
        model.embed_content(
            model='gemini-embedding-001',
            content=text,
            task_type=task_type
        )['embedding']
        for text in texts
    ]
```

#### 🔹 Step 2: Initialize and Configure Pinecone

```python
import pinecone

pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='us-west1-gcp')
index_name = 'gemini-index'
dimension = 768

if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=dimension, metric='cosine')

index = pinecone.Index(index_name)
```

#### 🔹 Step 3: Batch Insert Documents with Metadata

```python
documents = [
    {"id": "doc1", "text": "What is Gemini embedding?", "source": "FAQ"},
    {"id": "doc2", "text": "How does Pinecone work?", "source": "Docs"}
]

texts = [doc["text"] for doc in documents]
embeddings = get_embeddings(texts)

payload = [
    {
        "id": doc["id"],
        "values": emb,
        "metadata": {"source": doc["source"]}
    }
    for doc, emb in zip(documents, embeddings)
]

index.upsert(vectors=payload)
```

#### 🔹 Step 4: Query Pinecone with User Input

```python
def query_index(query_text, top_k=5):
    query_vector = get_embeddings([query_text], task_type="retrieval_query")[0]
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results

response = query_index("What is Gemini embedding?")
for match in response['matches']:
    print(f"Score: {match['score']:.4f}, Metadata: {match['metadata']}")
```

#### 🔹 Best Practices

```python
# Set consistent embedding dimension for index and queries
EMBEDDING_DIM = 768

from textwrap import wrap
# Chunk long documents into smaller parts for efficient embedding
CHUNK_SIZE = 300
chunks = wrap("Long document text here...", CHUNK_SIZE)

from tenacity import retry, stop_after_attempt, wait_fixed

# Add retry logic to handle transient API failures
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def safe_embed(text):
    return model.embed_content(
        model='gemini-embedding-001',
        content=text,
        task_type="retrieval_document"
    )["embedding"]

# Include rich metadata to enable filtering and faceted search
metadata = {
    "source": "Docs",
    "type": "tutorial",
    "language": "en",
    "tags": ["embedding", "gemini", "pinecone"]
}

# Use namespaces to logically isolate vector groups
NAMESPACE = "enterprise-docs"
index.upsert(vectors=payload, namespace=NAMESPACE)

# Monitor index usage metrics
stats = index.describe_index_stats()
print("Index size:", stats['total_vector_count'])

from dotenv import load_dotenv
import os
# Load API keys from environment variables
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")

# Cache embeddings to avoid redundant API calls and control cost
embedding_cache = {}
def cached_embed(text):
    if text in embedding_cache:
        return embedding_cache[text]
    embedding = safe_embed(text)
    embedding_cache[text] = embedding
    return embedding

# Re-embed documents periodically to prevent vector drift
import datetime
if datetime.datetime.now() - doc['last_updated'] > datetime.timedelta(days=7):
    updated_embedding = safe_embed(doc['text'])
```

## 💵 Pricing

| Tier      | Cost                       |
| --------- | -------------------------- |
| Free Tier | Explore via Gemini API     |
| Paid Tier | \$0.15 per 1M input tokens |

## 🔒 Security and Compliance

- Hosted on Google Cloud
- Vertex AI deployment supports:
  - SOC 2
  - HIPAA (with proper configuration)
  - Enterprise-grade governance

## 🛣️ Roadmap / Upcoming

- ⚙️ Batch API support for efficient large-scale embedding
- 🖼️ Multimodal embedding support planned (beyond text)

## 🧠 Summary

Gemini Embedding unlocks powerful capabilities for semantic understanding in over 100 languages. Its versatility across domains and scalability through Matryoshka Representation Learning makes it a preferred choice for modern AI pipelines.

Use it to supercharge your RAG systems, build intelligent search engines, automate classification, and create smarter AI-powered workflows.

## 📬 Need More?

Reach out for integration examples with:

- Pinecone, Weaviate, Qdrant
- LangChain for RAG
- Streamlit-based semantic search UIs

\#AI #Gemini #Embeddings #GoogleAI #LLM #SemanticSearch #VectorDB #RAG


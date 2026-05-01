# 🧠 RAG API System

A Retrieval-Augmented Generation (RAG) API that combines semantic search with large language models to generate accurate, context-aware answers.

---

##  Features

- Vector-based semantic retrieval (ChromaDB)
- LLM-powered answer generation
- FastAPI backend (high performance)
- Source-aware responses (RAG transparency)
- Modular architecture (easy to extend)

---

## Project Structure

```
project-root/
│
├── RAG/
│   ├── pipeline.py        # Core RAG logic (retrieve + generate)
│   ├── loader.py          # Data loading & preprocessing
│   └── chroma_db/         # Vector database (auto-generated)
│
├── app.py                 # FastAPI API server
│
└── README.md
```

---

## ⚙️ How It Works

```
User Question
    ↓
FastAPI (/chat endpoint)
    ↓
RAG Pipeline
   ├── Retrieve relevant chunks (ChromaDB)
   └── Generate answer (LLM)
    ↓
Return JSON response
```

---

## 🧪 API Usage

### POST `/chat`

#### Request
```json
{
  "question": "What is RAG?"
}
```

#### Response
```json
{
  "question": "What is RAG?",
  "answer": "RAG stands for Retrieval-Augmented Generation..."
}
```

---

## 🖥️ Setup & Installation

### 1. Clone repository
```bash
git clone <https://github.com/Lynn-Myat-Bhone/Easy-RAG.git>
cd <Easy-RAG>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### create .env file 
```
MODEL = gemini-2.5-flash
GEMINI_API_KEY = Your_GEMINI_API_KEY
EMBEDDING_MODEL=intfloat/multilingual-e5-base
CHROMA_DB_PATH=./chroma_db
TOP_K=4

DATASET_PATH = your_dataset_path

```

### 3. Run FastAPI server
```bash
uvicorn app:app --reload
```
---

## 📊 API Documentation

FastAPI provides interactive docs:
```
http://localhost:8000/docs
```

## Tech Stack

- Backend: FastAPI  
- Vector Database: ChromaDB  
- Embeddings: multilingual-e5-base
- LLM: API (Gemini, etc.)

---

## Example Use Cases

- Question Answering System  
- Knowledge Base Assistant  
- Document Search Engine  
- AI Chatbot Backend  

---

## Future Improvements

- Streaming responses (token-by-token output)
- Conversation memory
- Hybrid search (BM25 + vector)
- Reranking models
- Evaluation metrics (BERTScore, etc.)

---

## License

MIT License

---

## 👨‍💻 Author

Lynn Myat Bhone

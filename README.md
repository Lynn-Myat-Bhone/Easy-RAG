# Easy-RAG: Modular Retrieval-Augmented Generation API

A **Retrieval-Augmented Generation (RAG)** system designed for **question answering**, combining semantic retrieval with large language models to generate accurate, context-aware responses.

---

## Key Highlights

*  Focus on **Myanmar Dataset (low-resource language)**
*  Semantic retrieval using **multilingual E5 embeddings**
*  LLM-powered answer generation (Gemini API)
*  FastAPI backend for real-time inference
*  Evaluated using **BERTScore (F1: ~0.87)**

---

## Problem

Large Language Models often:

* Hallucinate answers
* Struggle with **low-resource languages like Myanmar**
* Lack access to domain-specific knowledge

---

## Solution

This project implements a **RAG pipeline**:

User Query → Retrieve relevant documents → Inject context → Generate grounded answer

---

## Architecture

```
User Question
    ↓
Embedding (multilingual-e5-base)
    ↓
ChromaDB Vector Search
    ↓
Top-K Relevant Context
    ↓
Prompt + Context Injection
    ↓
LLM (Gemini)
    ↓
Final Answer
```

---

## Evaluation

| Model             | Precision | Recall | F1 Score |
| ----------------- | --------- | ------ | -------- |
| RAG (This System) | 0.863     |0.882   | 0.872    |

RAG improves answer quality and grounding significantly.

---

## Project Structure

```
project-root/
│
├── RAG/
│   ├── pipeline.py        # Core RAG logic
│   ├── loader.py          # Data preprocessing
│   └── chroma_db/         # Vector database
│
├── app.py                 # FastAPI server
└── README.md
```

---

## ⚙️ How It Works

```
User Question
    ↓
FastAPI (/chat)
    ↓
Retrieve (ChromaDB)
    ↓
Generate (LLM)
    ↓
Return Answer
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

## Setup

```bash
git clone https://github.com/Lynn-Myat-Bhone/Easy-RAG.git
cd Easy-RAG
pip install -r requirements.txt
```

Create `.env`:

```
MODEL=gemini-2.5-flash
GEMINI_API_KEY=YOUR_KEY
EMBEDDING_MODEL=intfloat/multilingual-e5-base
CHROMA_DB_PATH=./chroma_db
TOP_K=4
DATASET_PATH=your_dataset
```

Run:

```bash
uvicorn app:app --reload
```

---

## Tech Stack

* FastAPI
* ChromaDB
* multilingual-e5-base (embeddings)
* Gemini API (LLM)

---

## Key Contributions

* RAG pipeline design (retrieval + generation)
* Evaluation using BERTScore
* Modular API architecture

---

## Future Work

* Hybrid retrieval (BM25 + dense)
* Reranking models
* Streaming responses
* Conversation memory

---

## Author

**Lynn Myat Bhone**

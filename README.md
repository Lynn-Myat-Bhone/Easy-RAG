# 🧠 RAG Chatbot with Streaming & Contextual Memory

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with:

* ChromaDB (vector database)
* HuggingFace embeddings
* Gemini API (LLM)
* Streaming responses (token-by-token)
* Contextual memory (multi-turn conversation)

---

## 🚀 Features

* Multilingual (Burmese + English)
* Vector search with ChromaDB
* Streaming responses 
* Context-aware conversation memory
* Auto DB build & load
---

## Project Structure

```
RAG/
    │── pipeline.py        # Core RAG logic (retrieve + generate + stream)
    │── loader.py          # Load CSV / JSON data
    │── chroma_db/         # Vector database (auto-generated)
```

---

## ⚙️ Installation

```bash
git clone <your-repo>
cd RAG

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```
EMBEDDING_MODEL=intfloat/multilingual-e5-base
CHROMA_DB_PATH=./chroma_db
TOP_K=3

GEMINI_API_KEY=your_api_key_here
MODEL=gemini-1.5-flash
```

---
## How It Works

```
User Question
    ↓
Retrieve (ChromaDB)
    ↓
Context + Memory
    ↓
Gemini LLM (Streaming)
    ↓
Answer (token-by-token)
```

---

##  Run the Chatbot (CLI)

```bash
python pipeline.py
```

### Example

```
You: သင်္ကြန်ပိတ်ရက်ဘယ်လောက်လဲ
Bot: ၂၀၂၆ ခုနှစ်၊ ဧပြီလ ၁၃ မှ ၁၆ ရက်နေ့အထိ ဖြစ်ပါသည်။
```

---


## 🧠 Contextual Memory


---


## 🛠️ Future Improvements

* 🔸 Redis caching
* 🔸 Hybrid search (keyword + vector)
* 🔸 Reranking models
* 🔸 Web UI (React / Next.js)
* 🔸 Multi-user session memory

---

## Debugging

Enable retrieval debug:

```python
print(score, doc.page_content[:50])
```

---

## Known Limitations

* Burmese embedding quality may vary
* Memory is in-memory (not persistent)
* Requires good dataset quality

---


## Author

Lynn Myat Bhone

---

## 📄 License

MIT License

import os

from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import loader
from TTS.AIVOOV_client import AIVOOVClient


class RAG:
    def __init__(self):
        load_dotenv()
        self.model = os.getenv("EMBEDDING_MODEL")
        self.db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.top_k = int(os.getenv("TOP_K", 4))
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("MODEL", "gemini-2.5-flash")
        self.aivoov_voice_id = os.getenv("AIVOOV_VOICE_ID", "")

        self.client = genai.Client(api_key=self.gemini_key)
        self.embedding = HuggingFaceEmbeddings(model_name=self.model)
        self.db = None
        self.memory = []
        self.max_history = 10

    def build_db(self, documents):
        self.db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.db_path,
        )

    def load_db(self):
        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding,
        )

    def retrieve(self, question):
        results = self.db.similarity_search_with_score(question, k=self.top_k) # type: ignore
        return [doc.page_content for doc, _ in results]

    def update_memory(self, question, answer):
        self.memory.append({"q": question, "a": answer})
        if len(self.memory) > self.max_history:
            self.memory.pop(0)

    def get_memory_context(self):
        history = []
        for turn in self.memory:
            history.append(f"User: {turn['q']}")
            history.append(f"Assistant: {turn['a']}")
        return "\n".join(history).strip()

    def build_prompt(self, question, context_docs):
        context = "\n\n".join(context_docs)
        history = self.get_memory_context()
        return f"""
You are a helpful assistant.

Use conversation history if relevant.

Conversation History:
{history}

Knowledge Context:
{context}

Current Question:
{question}

Answer in Burmese.
""".strip()

    def ask(self, question):
        docs = self.retrieve(question)
        if not docs:
            return "မေးခွန်းအတွက်တိကျသောအဖြေမရှိပါ"

        prompt = self.build_prompt(question, docs)
        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
        )

        answer = getattr(response, "text", None) or ""
        if not answer:
            answer = "မေးခွန်းအတွက်တိကျသောအဖြေမရှိပါ"

        self.update_memory(question, answer)
        return answer

    def text_to_speech(self, text):
        if not self.aivoov_voice_id:
            return None

        tts_client = AIVOOVClient()
        return tts_client.create_tts(
            text=text,
            voice_id=self.aivoov_voice_id,
        )


if __name__ == "__main__":
    rag = RAG()

    if not os.path.exists(rag.db_path):
        dataset_path = os.getenv("DATASET_PATH", "../dataset/q&a-v1.csv.csv")
        documents = loader.load_csv(dataset_path)
        rag.build_db(documents)

    rag.load_db()

    print("------------RAG + GEMINI + AIVOOV TTS---------------------")
    print("Type /exit to quit")

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"/exit", "exit", "quit"}:
            break

        answer = rag.ask(q)
        print(f"Bot: {answer}")

        tts_result = rag.text_to_speech(answer)
        if tts_result is not None:
            print(f"TTS: {tts_result}")


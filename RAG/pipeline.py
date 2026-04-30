import RAG.loader as loader
import os
from dotenv import load_dotenv
from google import genai
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAG:
    def __init__(self):
       load_dotenv()
       self.model = os.getenv("EMBEDDING_MODEL")
       self.db_path = os.getenv("CHROMA_DB_PATH","./chroma_db")
       self.top_k = int(os.getenv("TOP_K",4))
       self.gemini_key = os.getenv("GEMINI_API_KEY")
       self.gemini_model = os.getenv("MODEL")
       self.dataset_path = os.getenv("DATASET_PATH")
       self.client = genai.Client(api_key=self.gemini_key)
       
       self.embedding = HuggingFaceEmbeddings(
            model_name=self.model
       )
       self.db = None
       self.memory = []
       self.max_history = 10 # keep last 5 turns
       
    def build_db(self,documents):
           self.db = Chroma.from_documents(
               documents,
               self.embedding,
               persist_directory=self.db_path
           )
        
    def load_db(self):
        if not self.dataset_path:
            raise ValueError("Dataset path not provided")
        # Case 1: DB exists
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            self.db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding
            )
            print("[INFO] DB loaded successfully")
            return

        # Case 2: DB missing → build it
        print("[WARN] DB not found. Building new DB...")
        documents = loader.load_csv(self.dataset_path)
        self.build_db(documents)

        print("[INFO] DB built and ready")
    def query(self, question):
        results = self.db.similarity_search(question, k=self.top_k)
        return results[0].page_content if results else "No answer found"
    
    def retrieve(self, question):
        results = self.db.similarity_search_with_score(question, k=5)
        
        #debug
        # for doc, score in results:
        #     print(f"[DEBUG] score={score}")
        #     print(doc.page_content)
        #     print("-----")

        return [doc.page_content for doc, _ in results]
    
    def update_memory(self, question, answer):
        self.memory.append({
            "q": question,
            "a": answer
        })

        # keep only recent history
        if len(self.memory) > self.max_history:
            self.memory.pop(0)
    
    def get_memory_context(self):
        history = ""

        for turn in self.memory:
            history += f"User: {turn['q']}\n"
            history += f"Assistant: {turn['a']}\n"

        return history.strip()

    def generate(self, question, context_docs):
        context = "\n\n".join(context_docs)
        history = self.get_memory_context()

        prompt = f"""
        You are a helpful and intelligent assistant.

        Rules:
        - If the provided context is useful, use it.
        - if you dont found answer is centext_docs, say "I dont have the answer yet"
        - If the context is NOT useful or incomplete, answer using your own knowledge.
        - Always respond naturally and clearly in Burmese.

        Conversation History:
        {history}

        Context:
        {context}

        Question:
        {question}

        Now give a helpful answer:
        """

        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=prompt
        )

        return response.text

    def ask(self, question):
        docs = self.retrieve(question)

        if not docs:
            answer = "မေးခွန်းအတွက်တိကျသောအဖြေမရှိပါ"
            self.update_memory(question, answer)
            return answer

        answer = self.generate(question, docs)
        self.update_memory(question, answer)

        return answer
   

if __name__ == "__main__":
    rag = RAG()
    if not os.path.exists(rag.db_path):
        documents = loader.load_csv("../dataset/q&a-v1.csv.csv")
        rag.build_db(documents)

    rag.load_db()

    docs = rag.db.get()
    # print(docs.keys())
    # print(len(docs["documents"]))
    print("------------RAG + GEMINI (Streaming)---------------------")

    while True:
        q = input("\nYou: ")

        print("Bot: ", end="", flush=True)

        for token in rag.ask_stream(q):
            print(token, end="", flush=True)

        print()


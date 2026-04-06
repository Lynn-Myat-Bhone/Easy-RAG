import loader
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
       
       self.client = genai.Client(api_key=self.gemini_key)
       
       self.embedding = HuggingFaceEmbeddings(
            model_name=self.model
       )
       self.db = None
       
    def build_db(self,documents):
           self.db = Chroma.from_documents(
               documents,
               self.embedding,
               persist_directory=self.db_path
           )
        
    def load_db(self):
            self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding
            )
    def query(self, question):
        results = self.db.similarity_search(question, k=self.top_k)
        return results[0].page_content if results else "No answer found"
    
    def retrieve(self, question):
        results = self.db.similarity_search_with_score(question, k=5)

        for doc, score in results:
            print(f"[DEBUG] score={score}")
            print(doc.page_content)
            print("-----")

        return [doc.page_content for doc, _ in results]

    def generate(self, question, context_docs):
                context = "\n\n".join(context_docs)

                prompt = f"""
        You are a helpful assistant. Answer using ONLY the context below.

        Context:
        {context}

        Question:
        {question}

        Answer in Burmese.
        """

                response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
            )

                return response.text

    def ask(self, question):
            if not self.db:
                raise ValueError("DB not loaded!")

            docs = self.retrieve(question)

            if not docs:
                return "No answer found"

            return self.generate(question, docs)


if __name__ == "__main__":
    rag = RAG()

    if not os.path.exists(rag.db_path):
        documents = loader.load_csv("../dataset/q&a-v1.csv.csv")
        rag.build_db(documents)
    rag.load_db()
    docs = rag.db.get()

    print(docs.keys())
    print(len(docs["documents"]))
    
    print("------------RAG + GEMINI---------------------")
    print(rag.ask("ရန်ကုန် ကွန်ပျူတာတက္ကသိုလ်ကဘယ်မှာရှိတာလဲ "))


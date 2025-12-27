from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

class VectorStore:
    def __init__(self, persist_directory=None):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Vector store initialized")

    def create_from_documents(self, chunks):
        print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
        
        if self.persist_directory:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # In-memory only
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
        
        print(f"✅ Vector store created")
        return self.vectorstore
    
    def load_existing(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("✅ Loaded existing vector store")
        return self.vectorstore
    
    def similarity_search(self, query, k=3):
        print(f"🔍 Searching for: '{query}'")
        results = self.vectorstore.similarity_search(query, k=k)
        print(f"✅ Found {len(results)} relevant chunks")
        return results

if __name__ == "__main__":
    from document_loader import load_pdf
    
    chunks = load_pdf("data/test.pdf")
    vs = VectorStore()
    vectorstore = vs.create_from_documents(chunks)
    
    results = vs.similarity_search("What is this document about?", k=2)
    
    print("\n" + "="*50)
    for i, doc in enumerate(results, 1):
        print(f"\n📄 Result {i}:")
        print(doc.page_content[:200]) 
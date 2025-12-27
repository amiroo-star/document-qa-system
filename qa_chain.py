from langchain_community.llms import Ollama
from vector_store import VectorStore
from dotenv import load_dotenv
import os

load_dotenv()

class QASystem:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        
        # Use FREE local Ollama instead of OpenAI
        self.llm = Ollama(model="llama3.2")
        
        print("✅ Q&A System initialized (using local Llama)")
    
    def ask(self, question):
        print(f"\n❓ Question: {question}")
        
        # Get relevant documents
        docs = self.vectorstore.similarity_search(question, k=3)
        
        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""Use the following context to answer the question.
If you don't know the answer, say "I don't have enough information."

Context: {context}

Question: {question}

Answer:"""
        
        # Get answer
        answer = self.llm.invoke(prompt)
        
        print(f"\n💡 Answer: {answer}")
        print(f"\n📚 Used {len(docs)} source chunks")
        
        return answer, docs

if __name__ == "__main__":
    from document_loader import load_pdf
    
    vs = VectorStore()
    try:
        vectorstore = vs.load_existing()
    except:
        print("Creating new vector store...")
        chunks = load_pdf("data/test.pdf")
        vectorstore = vs.create_from_documents(chunks)
    
    qa = QASystem(vectorstore)
    
    questions = [
        "What is this document about?",
        "Who are the authors?",
        "What technologies are being compared?"
    ]
    
    for q in questions:
        qa.ask(q)
        print("\n" + "="*60 + "\n")
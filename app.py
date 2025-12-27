import streamlit as st
from document_loader import load_pdf
from vector_store import VectorStore
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="📄 Document Q&A System", layout="wide")

st.title("📄 AI-Powered Document Q&A System")
st.markdown("Upload a PDF and ask questions about it!")

# Initialize LLM (using Groq - Fast & Free)
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

# Sidebar
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=['pdf'])
    
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        pdf_path = "data/uploaded.pdf"
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("✅ PDF uploaded!")
        
        if st.button("🔄 Process Document"):
            with st.spinner("Processing document..."):
                chunks = load_pdf(pdf_path)
                st.write(f"✅ Split into {len(chunks)} chunks")
                
                # DON'T PERSIST - CREATE FRESH EACH TIME
                vs = VectorStore(persist_directory=None)  # In-memory only
                vectorstore = vs.create_from_documents(chunks)
                
                st.session_state['vectorstore'] = vectorstore
                st.session_state['processed'] = True

# Main area
if 'processed' in st.session_state and st.session_state['processed']:
    st.header("💬 Ask Questions")
    
    question = st.text_input("Enter your question:", placeholder="What is this document about?")
    
    if st.button("🔍 Get Answer") and question:
        with st.spinner("Generating answer..."):
            try:
                # Get LLM
                llm = get_llm()
                
                # Search for relevant chunks
                results = st.session_state['vectorstore'].similarity_search(question, k=3)
                
                # Build context
                context = "\n\n".join([doc.page_content for doc in results])
                
                # Create prompt
                prompt = f"""Use the following context from a document to answer the question.
If you cannot find the answer in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer concisely and directly:"""
                
                # Get answer from LLM
                response = llm.invoke(prompt)
                answer = response.content
                                # Debug: Show what was retrieved
                with st.expander("🔍 Debug: What the system found", expanded=False):
                    st.write(f"Retrieved {len(results)} chunks:")
                    for i, doc in enumerate(results, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content[:300])
                        st.divider()
                # Display answer
                st.markdown("### 💡 Answer:")
                st.success(answer)
                
                # Show sources
                with st.expander("📚 View Source Chunks (Click to expand)"):
                    for i, doc in enumerate(results, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:400] + "...")
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure you've added GROQ_API_KEY to your .env file!")

else:
    st.info("👈 Upload a PDF in the sidebar to get started!")
    
    # Show demo
    st.markdown("---")
    st.markdown("### 🎯 How it works:")
    st.markdown("""
    1. **Upload** a PDF document
    2. **Process** it (creates embeddings)
    3. **Ask** natural language questions
    4. **Get** AI-powered answers with sources
    """)
    
    st.markdown("---")
    st.markdown("### ✨ Features:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- 🔍 Semantic search")
        st.markdown("- 🤖 AI-powered answers")
    with col2:
        st.markdown("- 📚 Source citations")
        st.markdown("- 🆓 Free to use")

st.markdown("---")
st.markdown("⚡ Powered by Groq (Llama 3.1) & Streamlit")
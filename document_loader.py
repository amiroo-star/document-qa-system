from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_pdf(pdf_path):
    """
    Load a PDF and split it into chunks
    """
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"✅ Loaded {len(pages)} pages from PDF")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(pages)
    
    print(f"✅ Split into {len(chunks)} chunks")
    
    return chunks

# Test function
if __name__ == "__main__":
    test_pdf = "data/test.pdf"
    
    if os.path.exists(test_pdf):
        chunks = load_pdf(test_pdf)
        print(f"\n📄 First chunk preview:")
        print(chunks[0].page_content[:300])
    else:
        print(f"⚠️ Please add a PDF to data/test.pdf to test")
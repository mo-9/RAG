import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Step 1: Load and process .md files
def load_markdown_documents(directory: str):
    """Load all .md files from a directory."""
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".md"):
            loader = UnstructuredMarkdownLoader(os.path.join(directory, file))
            documents.extend(loader.load())
    return documents


# Step 2: Split documents into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into manageable chunks."""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


# Step 3: Create FAISS vector store
def create_faiss_index(texts, model_name="all-MiniLM-L6-v2"):
    """Create a FAISS vector store from texts."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(texts, embeddings)


# Step 4: Initialize LLM
def initialize_llm(model_name="llama3.1"):
    """Initialize the LLM with Ollama."""
    return OllamaLLM(model=model_name)


# Step 5: Build RAG system
def build_rag_system(llm, retriever):
    """Build a Retrieval-Augmented Generation (RAG) system."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


# Step 6: Streamlit application
def run_app():
    """Run the Streamlit app."""
    st.title("RAG System with LLaMA 3.1")

    # Load and process documents only once
    md_directory = "data/"
    if not os.path.exists(md_directory):
        st.error("Markdown directory not found. Please ensure the 'data/' folder exists.")
        return

    documents = load_markdown_documents(md_directory)
    if not documents:
        st.error("No markdown files found in the directory.")
        return

    texts = split_documents(documents)

    # Create vector store and initialize retriever
    vectorstore = create_faiss_index(texts)
    retriever = vectorstore.as_retriever()

    # Initialize LLM and RAG system
    llm = initialize_llm()
    rag_chain = build_rag_system(llm, retriever)

    # Query interface
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Fetching answer..."):
            result = rag_chain({"query": query})
            st.write("### Answer:")
            st.write(result["result"])
            st.write("### Source Documents:")
            for doc in result["source_documents"]:
                st.write(f"Source: {doc.metadata['source']}")
                st.write(doc.page_content[:500])  # Display preview


# Run the application
if __name__ == "__main__":
    run_app()

import os
import streamlit as st
from document_parser import extract_text_from_pdf
from embed_store import embed_and_store, load_documents_and_index
from rag_pipeline import answer_query
from langchain.docstore.document import Document
from openai import OpenAI  # Make sure you have openai python SDK installed
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="Finance AI Assistant")
st.title("📊 Finance Portfolio AI Assistant")
st.sidebar.header("Upload your financial document")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully.")
    st.write("Now analyzing the document...")

    # Save uploaded file
    os.makedirs("data/user_docs", exist_ok=True)
    file_path = f"data/user_docs/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("File saved successfully!")

    # Step 1: Extract text
    raw_text = extract_text_from_pdf(file_path)

    # Chunk text and wrap in Document objects
    text_chunks = raw_text.split("\n\n")  # Simple paragraph split
    documents = [Document(page_content=chunk) for chunk in text_chunks if chunk.strip()]

    # Step 2: Embed and store
    embed_and_store(text_chunks, documents)

# Load index and documents to answer queries
try:
    index, all_documents = load_documents_and_index()
except FileNotFoundError:
    st.warning("Please upload and embed documents first.")
    index, all_documents = None, []

# Initialize OpenAI client (make sure your environment variable OPENAI_API_KEY is set)
openai_client = OpenAI()

# Chat interface
query = st.text_input("Ask a question about your document:")
if query and index and documents:
    response = answer_query(query, index, documents, openai_client)
    st.write("🧠", response)

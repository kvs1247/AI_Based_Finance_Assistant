
'''
import os
import faiss
import pickle
import numpy as np
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # You can replace this with any LLM integration

#load_dotenv()

# Load embedding model
embedding_model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)

# Paths
INDEX_PATH = "data/faiss_index.index"
DOCS_PATH = "data/docs.pkl"

# Create the folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load or create FAISS index
def load_or_create_faiss_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        print("✅ Loading existing FAISS index and docs...")
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
    else:
        print("⚠️ No index found. Creating new one...")
        index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        documents = []
        faiss.write_index(index, INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)
    return index, documents

# Add new documents to index
def add_documents_to_index(texts: List[str], documents: List[Document], index):
    embeddings = embedding_model.encode(texts)
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"✅ Added {len(texts)} documents to FAISS index.")

# Simple query answering with LLM integration

import tiktoken

def num_tokens_from_string(string: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))

def answer_query(query: str, index, documents: List[Document], openai_client: OpenAI, top_k: int = 5, max_tokens_context: int = 9000):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]

    context_parts = []
    total_tokens = 0
    for doc in retrieved_docs:
        content = doc.page_content
        tokens = num_tokens_from_string(content)
        if total_tokens + tokens > max_tokens_context:
            break
        context_parts.append(content)
        total_tokens += tokens

    context = "\n\n".join(context_parts)
    prompt = f"You are a financial assistant. Use the context below to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    print("🧠 LLM Prompt (first 300 chars):", prompt[:300])

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant for financial portfolio analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()




# Example Document structure for testing
class DummyDocument:
    def __init__(self, content):
        self.page_content = content

# Optional: Test run
if __name__ == "__main__":
    from openai import OpenAI
    import os
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    index, docs = load_or_create_faiss_index()
    if not docs:
        sample_texts = ["Apple revenue increased this quarter", "Tesla faces supply chain issues"]
        sample_docs = [DummyDocument(text) for text in sample_texts]
        add_documents_to_index(sample_texts, sample_docs, index)

    response = answer_query("How is Apple performing?", index, docs, client)
    print("Response:", response)
'''

import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)

def num_tokens_from_string(string: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))

def answer_query(query: str, index, documents, openai_client, top_k: int = 5, max_tokens_context: int = 9000):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
    context_parts = []
    total_tokens = 0
    for doc in retrieved_docs:
        content = doc.page_content
        tokens = num_tokens_from_string(content)
        if total_tokens + tokens > max_tokens_context:
            break
        context_parts.append(content)
        total_tokens += tokens
    context = "\n\n".join(context_parts)
    prompt = f"You are a financial assistant. Use the context below to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant for financial portfolio analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

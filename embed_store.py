'''''''''
import os
import faiss
import pickle
import numpy as np
from typing import List
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

embedding_model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)

INDEX_PATH = "data/faiss_index.index"
DOCS_PATH = "data/docs.pkl"

os.makedirs("data", exist_ok=True)

def embed_and_store(texts: List[str], documents: List[Document]):
    emb_dim = embedding_model.get_sentence_embedding_dimension()

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        if index.d != emb_dim:
            print(f"⚠️ Existing FAISS index dimension {index.d} does not match model embedding dimension {emb_dim}. Recreating index.")
            index = faiss.IndexFlatL2(emb_dim)
    else:
        index = faiss.IndexFlatL2(emb_dim)

    embeddings = embedding_model.encode(texts)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)  # Ensure 2D array

    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)

    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "rb") as f:
            existing_docs = pickle.load(f)
        documents = existing_docs + documents

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"✅ Stored {len(texts)} new documents to FAISS index.")


def load_documents_and_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("FAISS index or documents file not found.")

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    return index, documents

'''

import os
import faiss
import pickle
import numpy as np
from typing import List
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

embedding_model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)

INDEX_PATH = "data/faiss_index.index"
DOCS_PATH = "data/docs.pkl"

os.makedirs("data", exist_ok=True)

def get_company_from_filename(filename):
    # Simple heuristic: use the first part of filename as company name
    return filename.split("_")[0].replace(".pdf", "")

def embed_and_store(texts: List[str], documents: List[Document]):
    emb_dim = embedding_model.get_sentence_embedding_dimension()
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        if index.d != emb_dim:
            print(f"⚠️ Existing FAISS index dimension {index.d} does not match model embedding dimension {emb_dim}. Recreating index.")
            index = faiss.IndexFlatL2(emb_dim)
    else:
        index = faiss.IndexFlatL2(emb_dim)
    embeddings = embedding_model.encode(texts)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_PATH)
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "rb") as f:
            existing_docs = pickle.load(f)
        documents = existing_docs + documents
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"✅ Stored {len(texts)} new documents to FAISS index.")

def load_documents_and_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("FAISS index or documents file not found.")
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    return index, documents

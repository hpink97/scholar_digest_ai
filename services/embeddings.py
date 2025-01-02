# services/embeddings.py

import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    raise RuntimeError("pysqlite3-binary is not installed. Add it to your requirements.txt.")

import chromadb
import streamlit as st
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

from sentence_transformers import SentenceTransformer
from services.etl import get_text_chunks

def init_chroma_db(collection_name: str = "biorxiv", persist_directory: str = "/tmp/.chroma") -> tuple:
    """
    Initialize Chroma DB connection using ChromaDBConnection and create a collection if it doesn't exist.
    
    :param collection_name: The name of the collection to create or connect to.
    :param persist_directory: Directory to persist the Chroma DB.
    :return: Tuple of (ChromaDBConnection, collection name)
    """
    configuration = {
        "client": "PersistentClient",
        "path": persist_directory
    }

    # Initialize connection to ChromaDB
    conn = st.connection(name="chromadb", type=ChromadbConnection, **configuration)

    # Create or get the collection
    conn.create_collection(
        collection_name=collection_name,
        embedding_function_name="",
        embedding_config={}
    )

    return conn, collection_name

def add_embeddings_to_db(data: dict, embeddings_model: SentenceTransformer, conn, collection_name: str):
    """Upsert document chunks into the Chroma DB."""
    if data is None:
        return
    if len(data.get("chunks", [])) == 0:
        return

    embeddings = embeddings_model.encode(data["chunks"], show_progress_bar=False)
    title = data["title"]
    doi = data["doi"]

    documents = []
    metadatas = []
    ids = []

    for i, (text, embedding) in enumerate(zip(data["chunks"], embeddings)):
        documents.append(text)
        metadatas.append({"id": i, "title": title, "doi": doi})
        ids.append(f"{i}_{doi}")

    conn.upload_documents(
        collection_name=collection_name,
        documents=documents,
        metadatas=metadatas,
        embeddings=[embedding.tolist() for embedding in embeddings],
        ids=ids
    )

def search_database(query: str, model: SentenceTransformer, conn, collection_name: str, top_k: int = 25) -> str:
    """Perform semantic search on the vector database."""
    query_embedding = model.encode(query)
    results = conn.query(
        collection_name=collection_name, 
        query_embedding=query_embedding.tolist(),
        num_results_limit=top_k,
        attributes=["documents"]
    )

    if results is not None and "documents" in results and results["documents"]:
        return "\n".join(results["documents"][0])
    else:
        return ""

def add_doi_embeddings(doi: str, embeddings_model: SentenceTransformer, conn, collection_name: str):
    """
    Extract chunks from a biorxiv paper and embed them into Chroma DB.
    `etl_service` is a reference to your ETL module (to avoid circular imports).
    """
    data = get_text_chunks(doi)
    add_embeddings_to_db(data, embeddings_model, conn, collection_name)
    return data

# services/embeddings.py

import chromadb
from chromadb import Client
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
from services.etl import get_biorxiv_chunks


def init_chroma_db(persist_directory: str = "./chroma_db") -> tuple:
    """Initialize Chroma DB and return (client, collection)."""
    chroma_db_client = Client(Settings(
        persist_directory=persist_directory,
        anonymized_telemetry=False
    ))
    collection = chroma_db_client.get_or_create_collection(name="biorxiv")
    return chroma_db_client, collection


def add_embeddings_to_db(data: dict, embeddings_model: SentenceTransformer, collection):
    """Upsert document chunks into the Chroma DB."""
    embeddings = embeddings_model.encode(data["chunks"], show_progress_bar=False)
    title = data["title"]
    doi = data["doi"]

    for i, (text, embedding) in enumerate(zip(data["chunks"], embeddings)):
        collection.upsert(
            ids=[f"{i}_{doi}"],
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[{"id": i, "title": title, "doi": doi}],
        )


def search_database(query: str, model: SentenceTransformer, collection, top_k: int = 25) -> str:
    """Perform semantic search on the vector database."""
    query_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    if results and "documents" in results and results["documents"]:
        return "\n".join(results["documents"][0])
    else:
        return ""


def process_biorxiv(doi: str, embeddings_model: SentenceTransformer, collection):
    """
    Extract chunks from a biorxiv paper and embed them into Chroma DB.
    `etl_service` is a reference to your ETL module (to avoid circular imports).
    """
    data = get_biorxiv_chunks(doi)
    add_embeddings_to_db(data, embeddings_model, collection)
    return data

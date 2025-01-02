# services/embeddings.py

import numpy as np
from sentence_transformers import SentenceTransformer
from services.etl import get_text_chunks

# In-memory storage for embeddings and metadata
class InMemoryVectorDB:
    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.metadata = []
        self.n_docs = 0

    def add(self, embedding, document, metadata):
        self.embeddings.append(embedding)
        self.documents.append(document)
        self.metadata.append(metadata)
        self.n_docs += 1

    def search(self, query_embedding, top_k):
        if not self.embeddings:
            return []

        # Calculate cosine similarity
        embeddings_array = np.array(self.embeddings)
        query_embedding = np.array(query_embedding)
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [
            {
                "title": self.metadata[i]["title"],
                "id": self.metadata[i]["id"],
                "document": self.documents[i],
            }
            for i in top_indices
        ]
        # sort by title/id
        sorted_results = sorted(results, key=lambda x: (x["title"], x["id"]))
        return [x["document"] for x in sorted_results]

def init_in_memory_db() -> InMemoryVectorDB:
    """Initialize an in-memory vector database."""
    return InMemoryVectorDB()

def add_embeddings_to_db(data: dict, embeddings_model: SentenceTransformer, db: InMemoryVectorDB):
    """Add document chunks and their embeddings to the in-memory DB."""
    if data is None:
        return
    if len(data.get("chunks", [])) == 0:
        return

    embeddings = embeddings_model.encode(data["chunks"], show_progress_bar=False)
    title = data["title"]
    doi = data["doi"]

    for i, (text, embedding) in enumerate(zip(data["chunks"], embeddings)):
        db.add(embedding, text, {"id": i, "title": title, "doi": doi})

def search_database(query: str, model: SentenceTransformer, db: InMemoryVectorDB, top_k: int = 10) -> list:
    """Perform a cosine similarity search on the in-memory DB."""
    query_embedding = model.encode(query)
    results = db.search(query_embedding, top_k=top_k)
    return results

def add_doi_embeddings(doi: str, embeddings_model: SentenceTransformer, db: InMemoryVectorDB):
    """
    Extract chunks from a biorxiv paper and embed them into the in-memory DB.
    """
    data = get_text_chunks(doi)
    add_embeddings_to_db(data, embeddings_model, db)
    return data

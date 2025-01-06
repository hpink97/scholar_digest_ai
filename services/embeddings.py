# services/embeddings.py

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
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
            print("No embeddings in the database.")
            return []

        # Calculate cosine similarity
        
        embeddings_array = np.array(self.embeddings)
        query_embedding = np.array(query_embedding).squeeze()
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
                "similarity": similarities[i],
            }
            for i in top_indices
        ]
        # sort by title/id
        return [x["document"] for x in results]

def init_in_memory_db() -> InMemoryVectorDB:
    """Initialize an in-memory vector database."""
    return InMemoryVectorDB()


def calculate_embeddings(text_list, embeddings_model, tokenizer):
    """Calculate embeddings for a list of texts."""
    # Tokenize input texts
    inputs = tokenizer(text_list, padding=True, return_tensors="pt")

    # Pass inputs through the model
    with torch.no_grad():
        outputs = embeddings_model(**inputs)

    # Extract last hidden states
    last_hidden_states = outputs.last_hidden_state

    # Mean-pooling
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask

    return sentence_embeddings.tolist()

def add_embeddings_to_db(data: dict, embeddings_model, tokenizer, db: InMemoryVectorDB):
    """Add document chunks and their embeddings to the in-memory DB."""
    if data is None:
        print("No data to add to the DB.")
        return
    if len(data.get("chunks", [])) == 0:
        print("No text chunks to add to the DB.")
        return

    embeddings = calculate_embeddings(data["chunks"], embeddings_model, tokenizer)
    title = data["title"]
    doi = data["doi"]

    for i, (text, embedding) in enumerate(zip(data["chunks"], embeddings)):
        db.add(embedding, text, {"id": i, "title": title, "doi": doi})


def search_database(query: str, embedding_model, tokenizer, db: InMemoryVectorDB, top_k: int = 10) -> list:
    """Perform a cosine similarity search on the in-memory DB."""
    query_embedding = calculate_embeddings([query], embedding_model, tokenizer)
    results = db.search(query_embedding, top_k=top_k)
    return results

def add_doi_embeddings(doi: str, embedding_model, tokenizer , db: InMemoryVectorDB):
    """
    Extract chunks from a biorxiv paper and embed them into the in-memory DB.
    """
    data = get_text_chunks(doi)
    add_embeddings_to_db(data, embedding_model, tokenizer, db)
    return data

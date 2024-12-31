# services/etl.py

import re
import requests
import pdfplumber
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_biorxiv_suffix(raw_doi: str) -> str:
    """Strip known prefixes like https://doi.org/ or doi:."""
    raw_doi = raw_doi.strip().lower()
    doi = re.sub(r'^https?://doi\.org/', '', raw_doi)
    return re.sub(r'^doi:', '', doi)


def get_biorxiv_metadata(raw_doi: str) -> dict:
    """Fetch metadata from BioRxiv for the given DOI."""
    doi_suffix = extract_biorxiv_suffix(raw_doi)
    url = f"https://api.biorxiv.org/details/biorxiv/{doi_suffix}/na/json"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def construct_pdf_url(metadata: dict) -> str:
    """Given the metadata dictionary, constructs the PDF URL."""
    base_url = "https://www.biorxiv.org/content"
    doi = metadata.get("collection", {})[-1].get("doi", "")
    version = metadata.get("collection", {})[-1].get("version", "1")
    return f"{base_url}/{doi}v{version}.full.pdf"


def extract_pdf_text(doi: str, tmp_path: str = "temp.pdf") -> dict:
    """Fetch PDF from BioRxiv, extract text, then remove temp file."""
    metadata = get_biorxiv_metadata(doi)
    title = metadata.get("collection", {})[-1].get("title", "")
    pdf_url = construct_pdf_url(metadata)

    resp = requests.get(pdf_url)
    resp.raise_for_status()
    if resp.headers.get("Content-Type") != "application/pdf":
        return {"text": "", "url": pdf_url, "doi": "", "title": ""}

    with open(tmp_path, "wb") as f:
        f.write(resp.content)

    all_text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text()

    os.remove(tmp_path)

    return {
        "text": all_text,
        "url": pdf_url,
        "doi": metadata.get("collection", {})[0].get("doi", ""),
        "title": title
    }


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks


def get_biorxiv_chunks(doi: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> dict:
    """Extract PDF text and split into chunks."""
    data = extract_pdf_text(doi)
    chunks = chunk_text(
        text=data["text"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    data["chunks"] = [i.replace("\n", " ").strip() for i in chunks]
    return data

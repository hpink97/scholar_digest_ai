# services/etl.py

import re
import requests
import pdfplumber
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _extract_doi_suffix(raw_doi: str) -> str:
    """Strip known prefixes like https://doi.org/ or doi:."""
    raw_doi = raw_doi.strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", raw_doi)
    return re.sub(r"^doi:", "", doi)


def doi_to_pmcid(doi):
    """Retrieve PMCID from Europe PMC by DOI."""
    doi = _extract_doi_suffix(doi)
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&format=json"
    r = requests.get(url)
    data = r.json()
    papers = data["resultList"]["result"]
    if len(papers) >= 1 and "pmcid" in papers[0]:
        return {
            "pmcid": papers[0]["pmcid"],
            "title": papers[0].get("title", "Unknown title"),
            "authors": papers[0].get("authorString", "Unknown authors"),
        }
    else:
        return None


def get_europe_pmc_fulltext(pmcid):
    """Retrieve full text (XML) from Europe PMC by PMCID."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    r = requests.get(url)
    if r.status_code == 200:
        return r.text  # Returns XML as a string
    else:
        print(f"Failed to retrieve full text for {pmcid}, status code: {r.status_code}")
        return None


def extract_article_text(jats_xml: str, body_only=True) -> dict:
    soup = BeautifulSoup(jats_xml, "lxml")

    # 1. Remove metadata elements you don't want.
    #    Examples: front, permissions, license, contrib-group, publisher, etc.
    for tag_name in [
        "front",  # Usually contains journal meta, authors, publisher, etc.
        "permissions",  # Licenses/copyright
        "license",  # Detailed licensing info
        "contrib-group",  # Author affiliations
        "funding-group",  # Funding info
        "publisher",  # Publisher info
    ]:
        unwanted_tags = soup.find_all(tag_name)
        for t in unwanted_tags:
            t.decompose()

    if body_only:
        # 2. Extract the main body text
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True)


def extract_text_pubmed(doi: str) -> dict:
    """Extract full text from Europe PMC by DOI."""
    data = doi_to_pmcid(doi)
    if not data or "pmcid" not in data:
        return None

    full_text = get_europe_pmc_fulltext(data["pmcid"])
    if not full_text:
        return None

    text = extract_article_text(full_text)
    if not text or len(text) < 100:
        return None
    return {"text": text, "doi": doi, "title": data["title"], "authors": data["authors"]}


def get_biorxiv_metadata(raw_doi: str) -> dict:
    """Fetch metadata from BioRxiv for the given DOI."""
    doi_suffix = _extract_doi_suffix(raw_doi)
    url = f"https://api.biorxiv.org/details/biorxiv/{doi_suffix}/na/json"
    resp = requests.get(url)
    # did it return a 200 with a json that contains "collection" key?
    if resp.status_code != 200:
        return None
    json_response = resp.json()
    if "collection" not in json_response or len(json_response.get("collection", [])) < 1:
        return None
    return json_response


def construct_biorxiv_pdf_url(metadata: dict) -> str:
    """Given the metadata dictionary, constructs the PDF URL."""
    base_url = "https://www.biorxiv.org/content"
    doi = metadata.get("collection", {})[-1].get("doi", "")
    version = metadata.get("collection", {})[-1].get("version", "1")
    return f"{base_url}/{doi}v{version}.full.pdf"


def extract_biorxiv_pdf_text(doi: str, tmp_path: str = "temp.pdf") -> dict:
    """Fetch PDF from BioRxiv, extract text, then remove temp file."""
    metadata = get_biorxiv_metadata(doi)
    if not metadata:
        return None
    pdf_url = construct_biorxiv_pdf_url(metadata)

    resp = requests.get(pdf_url)
    if resp.headers.get("Content-Type") != "application/pdf" or resp.status_code != 200:
        return None

    with open(tmp_path, "wb") as f:
        f.write(resp.content)

    all_text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text()

    os.remove(tmp_path)

    if len(all_text) < 100:
        return None

    return {
        "text": all_text,
        "doi": metadata.get("collection", {})[0].get("doi", ""),
        "title": metadata.get("collection", {})[-1].get("title", ""),
    }


def extract_doi_text(doi: str) -> dict:
    """Extract text from PDF using BioRxiv or Europe PMC."""
    biorxiv_text = extract_biorxiv_pdf_text(doi)
    if biorxiv_text is not None:
        return biorxiv_text

    pubmed_text = extract_text_pubmed(doi)
    if pubmed_text is not None:
        return pubmed_text

    return None


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


def get_text_chunks(doi: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> dict:
    """Extract PDF text and split into chunks."""
    data = extract_doi_text(doi)
    if data is None:
        return None

    chunks = chunk_text(text=data["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    data["chunks"] = [i.replace("\n", " ").strip() for i in chunks]
    return data


# ScholarDigestAI – Academic Paper Explainer

[**Launch the App**](https://scholar-digest-ai.streamlit.app/)

ScholarDigestAI is a Streamlit-based application that helps users explore and understand academic papers from a variety of sources. Users can:

1. **Provide article links** from repositories like [BioRxiv](https://www.biorxiv.org/), [arXiv](https://arxiv.org/), and [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/).
2. **Upload their own PDF files** for processing.
3. **Ask questions** or request specific content (e.g., summaries) tailored to different technical levels (elementary, high school, undergrad, domain expert).
4. **Optionally select** an LLM model (e.g., [Google Gemini](https://cloud.google.com/gemini) or [Meta Llama](https://ai.meta.com/llama/)) to generate responses.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)

---

## Features

- **DOI and URL Analysis**: Fetches the entire article text (including PDFs) from BioRxiv, arXiv, PubMed Central, or user-uploaded PDFs.
- **Multi-Source Support**: Supports academic repositories and uploaded documents, making it versatile for users working with various content sources.
- **Semantic Search and Summaries**: Embeds content using [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) for semantic search, retrieving relevant sections to answer user queries.
- **Customizable Technical Levels**: Offers tailored explanations for elementary, high school, undergraduate, or domain expert audiences.
- **Customizable Output Format**: Offers a variety of outputs from TL;DR to detailed summaries. 
- **Flexible LLM Integration**: Easily switch between different LLMs (e.g., Google Gemini, Meta Llama) to compare outputs.

---

## Project Structure

```
scholar_digest_ai/
├── services/
│   ├── __init__.py
│   ├── etl.py           # Extract/transform/load logic for article text and PDFs
│   ├── embeddings.py    # Embedding and semantic search logic
│   └── llm.py           # LLM client and prompt-building logic
├── app.py               # Streamlit application entry point
├── .streamlit/          # Streamlit configuration (theme, etc.)
├── requirements.txt     # Dependency requirements
└── README.md            # This readme
```

### Key Files

- **`app.py`**:  
  Launches the Streamlit app. Handles user inputs, processes articles, embeds data, and queries the LLM.
  
- **`services/etl.py`**:  
  Extracts text from DOIs, URLs, or uploaded PDFs. Splits large documents into smaller chunks for efficient processing.

- **`services/embeddings.py`**:  
  Creates and manages embeddings using [Sentence Transformers](https://www.sbert.net/), enabling semantic search.

- **`services/llm.py`**:  
  Integrates LLMs and manages prompts for different question formats and technical levels.

---

## Usage

### Running the App Locally

From the project root directory, run:

```bash
streamlit run app.py
```


### Interacting with the App

1. **Choose your input method**:
   - Enter a DOI or URL (e.g., from BioRxiv, arXiv, or PubMed Central).
   - Upload a PDF file.
2. **Ask a question** or request a summary.
3. **Select a technical level** (elementary, high school, undergrad, domain expert).
4. **Choose an LLM model** from the dropdown (e.g., `meta-llama/llama-3.3-70b-instruct`).
5. Click **Generate Answer** and review the response.

#### Single Document Mode
When one DOI, URL, or PDF is provided, all queries are answered using the extracted or uploaded document.

#### Multiple Document Mode
When multiple sources are provided, each is embedded into a Chroma DB. Relevant chunks are retrieved to answer the user’s question.

---

## Environment Variables

To enable access to external services, you may need to set the following environment variables:

- **`OPENROUTER_API_KEY`**: Required for LLM integration via [OpenRouter](https://openrouter.ai/).

Set them in your environment or in a `.env` file. Example:

```bash
# .env
OPENROUTER_API_KEY=sk-XXXXXXXXXX
```



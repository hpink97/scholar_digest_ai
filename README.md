
# ScholarDigestAI – BioRxiv Paper Explainer

ScholarDigestAI is a Streamlit-based application that fetches and explains preprints from [BioRxiv](https://www.biorxiv.org/). Users can:
1. **Provide one or more BioRxiv DOIs** for the application to process.
2. **Ask questions** or **request specific content** (e.g., summaries) at different technical levels (elementary, high school, undergrad, domain expert).
3. **Optionally select** an LLM model (e.g., [Google Gemini](https://cloud.google.com/gemini) or [Meta Llama](https://ai.meta.com/llama/)) to generate responses.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Customization](#customization)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Single DOI Analysis**: Fetches the entire paper PDF, extracts text, and allows direct question-answer interactions about that paper.
- **Multi-DOI Semantic Search**: Embeds multiple papers using [Sentence Transformers](https://www.sbert.net/) and stores them in a [Chroma DB](https://docs.trychroma.com/). Relevant chunks are retrieved to answer user queries.
- **Technical Level Control**: The user can request an explanation suitable for an elementary-level student, a high school student, an undergrad, or a domain expert.
- **LLM Model Selection**: Easily switch between different LLMs (e.g., Google Gemini, Meta Llama) to compare outputs.

---

## Project Structure

```
my_project_name/
├── services/
│   ├── __init__.py
│   ├── etl.py           # Extract/transform/load logic for BioRxiv PDFs
│   ├── embeddings.py    # Functions for embedding text and Chroma DB interactions
│   └── llm.py           # LLM client and prompt-building logic
├── app.py               # Streamlit application entry point
├── pyproject.toml       # Build system & dependency configuration, including Black formatting
└── README.md            # This readme
```

### Key Files

- **`app.py`**  
  Launches the Streamlit app. Handles user inputs, calls functions from `services/` to process DOIs, embed data, and query the LLM.
- **`services/etl.py`**  
  Functions for fetching PDFs from BioRxiv, extracting text, and splitting the text into smaller chunks.
- **`services/embeddings.py`**  
  Creates or loads a [Chroma DB](https://docs.trychroma.com/) collection, embeds text chunks using [Sentence Transformers](https://www.sbert.net/), and provides a semantic search method.
- **`services/llm.py`**  
  Houses the LLM (chat/completions) client, system prompt building, and a higher-level `explain_paper` function that organizes user/system messages for multi-turn dialogues or single responses.

---

## Usage

### Running the App

From the project root directory, run:
```bash
streamlit run app.py
```
This will start the Streamlit server. You can then open the displayed URL (usually `http://localhost:8501`) in your web browser.

### Interacting with the App

1. **Enter BioRxiv DOI(s)** in the left sidebar (one per line).
2. **Type your question** or request in the `Question / Request` text field.
3. **Select a technical level** (elementary, high school, undergrad, domain expert).
4. **Choose an LLM model** from the dropdown (e.g.,`meta-llama/llama-3.3-70b-instruct`).
5. Click **Generate Answer** and wait for the response to appear.

#### Single DOI Mode
If **one** DOI is provided, the entire text of that paper is fetched, and all queries are answered using that text directly.

#### Multiple DOI Mode
If **multiple** DOIs are provided, each is embedded into a Chroma DB. Relevant chunks from across all DOIs are retrieved to answer the user’s question.

---

## Environment Variables

Some services may require API keys or configuration values. For example:

- **`OPENROUTER_API_KEY`** for [OpenRouter](https://openrouter.ai/) usage.  
- Any other model-specific credentials (e.g., Hugging Face tokens).

Set them in your environment or in a `.env` file. For example:

```bash
# .env
OPENROUTER_API_KEY=sk-XXXXXXXXXX
```


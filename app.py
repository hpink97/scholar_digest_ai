import streamlit as st
import emoji
from transformers import AutoTokenizer, AutoModel
import io

from services.etl import (
    extract_doi_text,
    extract_text_from_uploaded_pdf,  # your helper for PDF
)
from services.embeddings import (
    init_in_memory_db,
    add_embeddings,
    search_database,
)
from services.llm import ScholarDigestAI

ROBOT_EMOJI = emoji.emojize(":robot:")
DATA_LOADING_EMOJI = emoji.emojize(":hourglass_flowing_sand:")
GREEN_CHECKMARK_EMOJI = emoji.emojize(":heavy_check_mark:")
LINK_EMOJI = emoji.emojize(":link:")
PDF_EMOJI = emoji.emojize(":page_facing_up:")
SETTINGS_EMOJI = emoji.emojize(":gear:")

# WORD LIMIT THRESHOLD
MAX_WORDS = 20_000


def _initialise_embeddings_model(model_name):
    return {
        "name": model_name,
        "tokenizer": AutoTokenizer.from_pretrained(model_name),
        "model": AutoModel.from_pretrained(model_name),
    }


def main():
    # -- Session State Vars --
    if "doi_list" not in st.session_state:
        st.session_state["doi_list"] = []
    if "previous_dois" not in st.session_state:
        st.session_state["previous_dois"] = []
    if "dataset_loaded" not in st.session_state:
        st.session_state["dataset_loaded"] = False
    if "full_text" not in st.session_state:
        st.session_state["full_text"] = ""
    if "title" not in st.session_state:
        st.session_state["title"] = ""
    if "scholar_ai" not in st.session_state:
        st.session_state["scholar_ai"] = None
    if "embeddings_db" not in st.session_state:
        st.session_state["embeddings_db"] = init_in_memory_db()
    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = None

    # Dark theme HTML content
    html_content = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    .styled-box {
      background-color: #2B2B2B; /* darker gray backdrop */
      border: 1px solid #444;    /* subtle border */
      border-radius: 8px;
      padding: 16px;
      margin-top: 10px;
      font-family: Arial, sans-serif;
    }

    .styled-box h2 {
      color: #fafafa;
      margin-top: 0;
      margin-bottom: 0.5em;
    }

    .styled-box p {
      font-size: 1.05em;
      line-height: 1.5;
      color: #f2f2f2;
      margin: 0.5em 0;
    }

    a {
      color: #56B4E9;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }

    /* Example for highlighting a specific word or phrase */
    .highlight {
      color: #EF8354;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <div class="styled-box">
    <h2>ScholarDigestAI</h2>
    <p>
      <strong>ScholarDigestAI</strong> is an <em>AI assistant</em> that helps you
      digest scientific papers.
    </p>
    <p>
    Enter an article <strong class="highlight">DOI</strong> from 
      <a href="https://www.biorxiv.org/" target="_blank">BioRxiv</a>,
      <a href="https://pmc.ncbi.nlm.nih.gov/" target="_blank">PubMed Central</a>, or
      <a href="https://arxiv.org/" target="_blank">arXiv</a>, or upload a PDF file.
    </p>
  </div>
</body>
</html>
"""
    st.html(html_content)

    # --- Radio Button: Select Input Type ---
    doi_option = f"Enter a DOI {LINK_EMOJI}"
    pdf_option = f"Upload a PDF {PDF_EMOJI}"
    input_mode = st.radio(
        "Choose your input method:", [doi_option, pdf_option], index=0
    )

    uploaded_pdf = None
    dois_input = ""

    if input_mode == doi_option:
        # DOI Input
        dois_input = st.text_area(
            "Enter DOI(s) or URLs, one per line:",
            value="https://doi.org/10.1101/2023.07.19.549542",
            height=65,
        )
    else:
        # PDF Upload
        uploaded_pdf = st.file_uploader(
            "Upload a PDF file", type=["pdf"], accept_multiple_files=False
        )

    # Question Input
    question_input = st.text_area(
        "Ask a question or request a summary:",
        value="Please provide a summary of the key findings, and discuss their implications.",
        height=65,
    )

    # Collapsible area for additional parameters
    with st.expander(f"Additional Settings {SETTINGS_EMOJI}"):
        techincal_level = st.selectbox(
            "Select Technical Level",
            [
                "elementary",
                "high school",
                "non-specialist",
                "undergrad",
                "domain expert",
            ],
            index=1,
        )
        model_choice = st.selectbox(
            "Select Model",
            [
                "meta-llama/llama-3.3-70b-instruct",
                "mistralai/ministral-8b",
                "google/gemini-flash-1.5",
                "deepseek/deepseek-chat",
                "openrouter/auto",
            ],
            index=2,
        )
        format_choice = st.selectbox(
            "How should the response be formatted?",
            ["TL;DR", "Concise Bullet Points", "Short summary", "Detailed summary"],
            index=1,
        )
        language_choice = st.text_input(
            "What language should the AI respond in?:", "English"
        )

    # Determine button label
    button_label = (
        "Ask AI" if not st.session_state["dataset_loaded"] else "Ask Another Question"
    )
    ask_btn = st.button(button_label)

    if ask_btn:
        # Branch based on input mode
        if input_mode == doi_option:
            # -- Handle DOI/URL logic --
            new_doi_list = [d.strip() for d in dois_input.split("\n") if d.strip()]
            if not new_doi_list:
                st.error("Please enter at least one DOI or URL.")
                return

            # Check if DOIs changed
            if new_doi_list != st.session_state["previous_dois"]:
                # Re-init data
                st.session_state["dataset_loaded"] = False
                st.session_state["embeddings_db"] = init_in_memory_db()
                st.session_state["previous_dois"] = new_doi_list
                st.session_state["doi_list"] = new_doi_list

                with st.spinner(f"{DATA_LOADING_EMOJI} Loading Data..."):
                    if len(new_doi_list) == 1:
                        # Single paper scenario
                        doc_data = extract_doi_text(new_doi_list[0])
                        if doc_data is None:
                            st.warning(f"Failed to fetch text for `{new_doi_list[0]}`.")
                            return

                        # Store text, get word count
                        st.session_state["full_text"] = doc_data.get("text", "")
                        word_count = len(st.session_state["full_text"].split())
                        st.session_state["title"] = doc_data.get("title", "")

                        # If the doc is bigger than MAX_WORDS => embed
                        if word_count > MAX_WORDS:
                            st.write(
                                f"Document has {word_count:,} words, exceeding {MAX_WORDS:,}. "
                                "Building embeddings for more efficient retrieval..."
                            )
                            if st.session_state["embeddings_model"] is None:
                                st.session_state["embeddings_model"] = (
                                    _initialise_embeddings_model(
                                        "NeuML/pubmedbert-base-embeddings"
                                    )
                                )
                            # Use the add_embeddings approach – but we adapt it for a single doc
                            added_data = add_embeddings(
                                doc_data,
                                st.session_state["embeddings_model"]["model"],
                                st.session_state["embeddings_model"]["tokenizer"],
                                st.session_state["embeddings_db"],
                            )
                            # Clear out the full_text, so we rely on embeddings
                            st.session_state["full_text"] = ""
                            st.session_state["scholar_ai"] = ScholarDigestAI(
                                article_text=""
                            )
                            st.success(
                                f"Generated embeddings for `{doc_data.get('title', '')}` "
                                f"({word_count:,} words)."
                            )
                        else:
                            # Smaller doc => pass entire text
                            st.session_state["scholar_ai"] = ScholarDigestAI(
                                article_text=st.session_state["full_text"]
                            )
                            st.success(
                                f"Fetched `{st.session_state['title']}` "
                                f"({word_count:,} words)."
                            )

                    else:
                        # Multiple papers => embeddings approach (unchanged)
                        if st.session_state["embeddings_model"] is None:
                            st.session_state["embeddings_model"] = (
                                _initialise_embeddings_model(
                                    "NeuML/pubmedbert-base-embeddings"
                                )
                            )
                        st.markdown(
                            f"Building embeddings for {len(new_doi_list)} papers "
                            f"with `{st.session_state['embeddings_model']['name']}`..."
                        )
                        for doi in new_doi_list:
                            single_data = extract_doi_text(doi)
                            added_data = add_embeddings(
                                single_data,
                                st.session_state["embeddings_model"]["model"],
                                st.session_state["embeddings_model"]["tokenizer"],
                                st.session_state["embeddings_db"],
                            )
                            if added_data is None:
                                st.warning(f"Failed to fetch text for `{doi}`.")
                            else:
                                st.success(
                                    f"Generated embeddings for `{added_data.get('title', '')}` "
                                    f"({len(added_data.get('text', '').split()):,} words)."
                                )
                        # No single large text
                        st.session_state["full_text"] = ""
                        st.session_state["title"] = "Multiple DOIs"
                        st.session_state["scholar_ai"] = ScholarDigestAI(
                            article_text=""
                        )

                    st.session_state["dataset_loaded"] = True
                    st.write(f"{GREEN_CHECKMARK_EMOJI} Data Loaded.")

        else:
            # -- Handle PDF logic --
            if not uploaded_pdf:
                st.error("Please upload a PDF file.")
                return

            if (
                "pdf_filename" not in st.session_state
                or st.session_state["pdf_filename"] != uploaded_pdf.name
            ):
                # Reload data only if a different PDF was uploaded
                st.session_state["pdf_filename"] = uploaded_pdf.name
                st.session_state["dataset_loaded"] = False
                st.session_state["embeddings_db"] = init_in_memory_db()
                st.session_state["doi_list"] = []
                st.session_state["previous_dois"] = []

                with st.spinner(f"{DATA_LOADING_EMOJI} Reading PDF..."):
                    pdf_text = extract_text_from_uploaded_pdf(uploaded_pdf)
                    word_count = len(pdf_text.split())
                    if word_count < 10:
                        st.warning("Failed to extract sufficient text from PDF.")
                        return

                    st.session_state["full_text"] = pdf_text
                    st.session_state["title"] = uploaded_pdf.name

                    # Decide if we embed based on size
                    if word_count > MAX_WORDS:
                        st.write(
                            f"PDF has {word_count:,} words, exceeding {MAX_WORDS:,}. "
                            "Building embeddings for more efficient retrieval..."
                        )
                        if st.session_state["embeddings_model"] is None:
                            st.session_state["embeddings_model"] = (
                                _initialise_embeddings_model(
                                    "NeuML/pubmedbert-base-embeddings"
                                )
                            )
                        # We'll adapt add_embeddings to handle a dictionary as well
                        add_dict = {
                            "text": pdf_text,
                            "title": uploaded_pdf.name,
                            "doi": uploaded_pdf.name,  # or some identifier
                        }
                        add_embeddings(
                            add_dict,
                            st.session_state["embeddings_model"]["model"],
                            st.session_state["embeddings_model"]["tokenizer"],
                            st.session_state["embeddings_db"],
                        )
                        # Clear out the large text
                        st.session_state["full_text"] = ""
                        st.session_state["scholar_ai"] = ScholarDigestAI(
                            article_text=""
                        )
                        st.success(
                            f"Generated embeddings for `{uploaded_pdf.name}` "
                            f"({word_count:,} words)."
                        )
                    else:
                        st.session_state["scholar_ai"] = ScholarDigestAI(
                            article_text=st.session_state["full_text"]
                        )
                        st.success(
                            f"Uploaded and extracted text from `{uploaded_pdf.name}`, "
                            f"({word_count:,} words)."
                        )
                st.session_state["dataset_loaded"] = True
                st.write(f"{GREEN_CHECKMARK_EMOJI} Data Loaded.")

        # --- Now ask the question ---
        if not st.session_state["dataset_loaded"]:
            st.error("Data couldn't be loaded. Check your input and retry.")
            return

        if not question_input.strip():
            st.error("Please enter a question.")
            return

        # If we either have multiple DOIs or a single doc above limit, we rely on embeddings
        relevant_sections = None
        # If there's anything in the embeddings DB, let's do a search for relevant chunks
        if (
            st.session_state["embeddings_db"].n_docs > 0
            and st.session_state["embeddings_model"]
        ):
            relevant_sections = search_database(
                query=question_input,
                embedding_model=st.session_state["embeddings_model"]["model"],
                tokenizer=st.session_state["embeddings_model"]["tokenizer"],
                db=st.session_state["embeddings_db"],
                top_k=min(st.session_state["embeddings_db"].n_docs, 20),
            )

        with st.spinner(f"{emoji.emojize(':robot:')} AI is thinking..."):
            response = st.session_state["scholar_ai"].ask_question(
                question=question_input,
                technical_level=techincal_level,
                format=format_choice,
                language=language_choice,
                model=model_choice,
                relevant_sections=relevant_sections,
            )

        st.markdown("### AI Response:")
        st.markdown(response)


if __name__ == "__main__":
    main()

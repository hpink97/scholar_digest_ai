# app.py

import streamlit as st
import emoji
from sentence_transformers import SentenceTransformer, models

# Import our custom services
from services.etl import extract_doi_text
from services.embeddings import init_chroma_db, add_doi_embeddings, search_database
from services.llm import ScholarDigestAI

ROBOT_EMOJI = emoji.emojize(":robot:")
FILING_CABINET_EMOJI = emoji.emojize(":card_file_box:")
DATA_LOADING_EMOJI = emoji.emojize(":hourglass_flowing_sand:")


# Initialize global variables
current_dois = []


def main():
    global scholar_ai, current_dois

    # ---- Initialize Embedding Model ----
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True
    )
    bio_clinical_bert = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # ---- Initialize or Load Chroma DB ----
    chroma_db_client, chroma_db_collection = init_chroma_db("./chroma_db")

    # ---- Streamlit UI ----
    st.title("ScholarDigestAI - Paper Explainer")
    st.sidebar.title("Load Paper Data")

    # Sidebar for loading data
    with st.sidebar:
        st.subheader("Paper(s) to Analyze")
        dois_input = st.text_area(
            "Enter DOI(s), one per line:",
            value="https://doi.org/10.1101/2023.07.19.549542\nhttps://doi.org/10.1007/s00122-022-04129-5",
        )
        load_data_btn = st.button("Load Data")

    # Main panel for asking questions
    st.subheader("Ask Questions About the Papers")
    question_input = st.text_area("Your Question:")
    techincal_level = st.selectbox(
        "Select Technical Level", ["elementary", "high school", "undergrad", "domain expert"], index=3
    )
    model_choice = st.selectbox(
        "Select Model",
        options=[
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/ministral-8b",
            "google/gemini-flash-1.5",
            "deepseek/deepseek-chat",
        ],
        index=0,
    )
    ask_btn = st.button("Ask AI")

    # ---- Handle Sidebar Button Click (Load Data) ----
    if load_data_btn:
        raw_dois = [d.strip() for d in dois_input.split("\n") if d.strip()]
        if not raw_dois:
            st.error("Please enter at least one DOI.")
            return

        current_dois = raw_dois
        st.session_state["dataset_loaded"] = False

        # Handle single or multiple DOIs
        with st.spinner(
            f"{FILING_CABINET_EMOJI} Loading and processing article data... {DATA_LOADING_EMOJI}"
        ):
            # message = ""
            for doi in raw_dois:
                if len(raw_dois) == 1:
                    doc_data = extract_doi_text(doi)
                    if doc_data is None:
                        st.warning(f"Failed to fetch text for `{doi}`. Check the DOI.\n")
                    else:
                        st.session_state["full_text"] = doc_data.get("text", "")
                        st.session_state["title"] = doc_data.get("title", "")
                        st.success(f"Fetched `{st.session_state['title']}` ({len(st.session_state['full_text'].split()):,} words).\n")
                else:
                    data = add_doi_embeddings(doi, bio_clinical_bert, chroma_db_collection)
                    if data is None:
                        st.warning(f"Failed to fetch text for `{doi}`.\n")
                    else:
                        st.success(f"Generated embeddings for `{data.get('title', '')}` ({len(data.get('text', '').split()):,} words).\n")

            st.session_state["dataset_loaded"] = True
            st.session_state["scholar_ai"] = ScholarDigestAI(
                article_text=st.session_state.get("full_text")
            )

    

    # ---- Handle Main Panel Button Click (Ask AI) ----
    if ask_btn:
        if not st.session_state.get("dataset_loaded", False):
            st.error("Please load data before asking questions.")
            return

        if not question_input.strip():
            st.error("Please enter a question.")
            return

        if len(current_dois) > 1:
            relevant_sections = search_database(
                query=question_input,
                model=bio_clinical_bert,
                collection=chroma_db_collection,
                top_k=min(int(chroma_db_collection.count() * 0.2), 15),
            )
        else:
            relevant_sections = None

        with st.spinner(f"{ROBOT_EMOJI} AI is thinking... {ROBOT_EMOJI}"):
            response = st.session_state["scholar_ai"].ask_question(
                question=question_input,
                technical_level=techincal_level,
                model=model_choice,
                relevant_sections=relevant_sections,
            )

        st.markdown(f"### AI Response:")
        st.markdown(response)


if __name__ == "__main__":
    main()

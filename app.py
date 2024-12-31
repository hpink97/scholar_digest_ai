# app.py

import streamlit as st
import emoji
from sentence_transformers import SentenceTransformer, models

# Import our custom services
from services.etl import extract_pdf_text
from services.embeddings import init_chroma_db, process_biorxiv, search_database
from services.llm import explain_paper

ROBOT_EMOJI = emoji.emojize(":robot:")

def main():
    # ---- 1. Initialize Embedding Model ----
    # For example: Bio_ClinicalBERT
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    bio_clinical_bert = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # ---- 2. Initialize or load Chroma DB ----
    chroma_db_client, chroma_db_collection = init_chroma_db("./chroma_db")

    st.title("ScholarDigestAI - BioRxiv Paper Explainer")
    instruction = st.empty()
    instruction.write(
        "Enter one or more BioRxiv DOI(s). Then type your question (or request) about the paper(s), "
        "choose the technical level, and select a model to generate a response."
    )

    # ---- Sidebar Inputs ----
    with st.sidebar:
        st.subheader("Paper(s) to Analyze")
        dois_input = st.text_area(
            "Enter BioRxiv DOI(s), one per line:",
            value="https://doi.org/10.1101/2023.07.19.549542"
        )

        st.subheader("Question / Request")
        user_question = st.text_area(
            "What do you want to ask or request?",
            value="Please provide a detailed summary of the paper."
        )

        techincal_level = st.selectbox(
            "Select Technical Level",
            ["elementary", "high school", "undergrad", "domain expert"],
            index=3
        )

        model_choice = st.selectbox(
            "Select Model",
            options=[
                "meta-llama/llama-3.3-70b-instruct",
                "mistralai/ministral-8b", 
                "google/gemini-flash-1.5",
                "deepseek/deepseek-chat"
            ],
            index=0
        )

        generate_btn = st.button("Generate Answer")

    # ---- Handle Button Click ----
    if generate_btn:
        raw_dois = [d.strip() for d in dois_input.split("\n") if d.strip()]
        if not raw_dois:
            st.error("Please enter at least one DOI.")
            return

        if not user_question.strip():
            st.error("Please enter your question/request.")
            return

        # If there's only one DOI, fetch the entire text
        if len(raw_dois) == 1:
            single_doi = raw_dois[0]
            
            # Create placeholders for temporary messages
            fetching_message = st.empty()            
            # Display fetching message
            fetching_message.write(f"Fetching {single_doi} ...")
            doc_data = extract_pdf_text(single_doi)
            full_text = doc_data.get("text", "")
            n_words = len(full_text.split())
            
            # Update fetching message
            fetching_message.write(f"Fetched text for {doc_data.get('title', '')} ({n_words} words)")
            
            # Display spinner message for processing
            with st.spinner(f"{ROBOT_EMOJI} AI is thinking... Please wait {ROBOT_EMOJI} "):
                response = explain_paper(
                    question=user_question,
                    model=model_choice,
                    techincal_level=techincal_level,
                    article_text=full_text
                )
            
                # Display the response
            fetching_message.empty()
            instruction.empty()
            st.markdown(response)


        else:
            # If multiple DOIs, let's embed all and rely on semantic search
            fetching_message = st.empty() 
            msg = f"Creating embeddings with `{model_name}`."
            fetching_message.write(msg)
            for doi in raw_dois:
                
                data = process_biorxiv(doi, bio_clinical_bert, chroma_db_collection )
                msg += f"\n\nFetched text and generated embeddings for `{data.get('title', '')}` ({len(data.get('text', '').split()):,} words)."
                fetching_message.write(msg)


            # Provide a partial function or lambda for the search
            k = int(chroma_db_collection.count()*0.15)
            def _search_fn(query, k=k):
                return search_database(query, bio_clinical_bert, chroma_db_collection, top_k=k)

            with st.spinner(f"{ROBOT_EMOJI} AI is thinking... Please wait {ROBOT_EMOJI}"):
                response = explain_paper(
                    question=user_question,
                    model=model_choice,
                    techincal_level=techincal_level,
                    article_text=None,
                    search_fn=_search_fn
                )
            fetching_message.empty()
            instruction.empty()
            st.markdown(response)


if __name__ == "__main__":
    main()

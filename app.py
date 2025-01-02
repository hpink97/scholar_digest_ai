import streamlit as st
import emoji
from sentence_transformers import SentenceTransformer, models

from services.etl import extract_doi_text
from services.embeddings import init_in_memory_db, add_doi_embeddings, search_database
from services.llm import ScholarDigestAI

ROBOT_EMOJI = emoji.emojize(":robot:")
FILING_CABINET_EMOJI = emoji.emojize(":card_file_box:")
DATA_LOADING_EMOJI = emoji.emojize(":hourglass_flowing_sand:")
BOOK_STACK_EMOJI = emoji.emojize(":books:")

def main():

    # -- Initialize Session State Vars if not already present --
    if "doi_list" not in st.session_state:
        st.session_state["doi_list"] = []
    if "dataset_loaded" not in st.session_state:
        st.session_state["dataset_loaded"] = False
    if "full_text" not in st.session_state:
        st.session_state["full_text"] = ""
    if "title" not in st.session_state:
        st.session_state["title"] = ""
    if "scholar_ai" not in st.session_state:
        st.session_state["scholar_ai"] = None
    # You may also want to store embeddings_db in session_state if you need it
    if "embeddings_db" not in st.session_state:
        st.session_state["embeddings_db"] = init_in_memory_db()

    # ---- Initialize Embedding Model ----
    model_name = "allenai/scibert_scivocab_cased"
    embedding_model = SentenceTransformer(model_name)

    # For readability:
    embeddings_db = st.session_state["embeddings_db"]

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

        if load_data_btn:
            # Parse the DOIs from user input, store them in session state
            doi_list = [d.strip() for d in dois_input.split("\n") if d.strip()]
            if len(doi_list) == 0:
                st.error("Please enter at least one DOI.")
                return

            st.session_state["doi_list"] = doi_list
            st.session_state["dataset_loaded"] = False


            st.markdown(f"{DATA_LOADING_EMOJI} Loading Data...")
            if len(doi_list) == 1:
                doc_data = extract_doi_text(doi_list[0])
                if doc_data is None:
                    st.warning(f"Failed to fetch text for `{doi_list[0]}`. Check the DOI.\n")
                else:
                    st.session_state["full_text"] = doc_data.get("text", "")
                    st.session_state["title"] = doc_data.get("title", "")
                    st.success(
                        f"Fetched `{st.session_state['title']}` "
                        f"({len(st.session_state['full_text'].split()):,} words).\n"
                    )
                    st.session_state["scholar_ai"] = ScholarDigestAI(
                        article_text=st.session_state["full_text"]
                    )
            else:
                # multiple DOIs -> build embeddings
                st.markdown(f"Building embeddings for {len(doi_list)} papers with {model_name}...")
                for doi in doi_list:
                    data = add_doi_embeddings(doi, embedding_model, embeddings_db)
                    if data is None:
                        st.warning(f"Failed to fetch text for `{doi}`.\n")
                    else:
                        st.success(
                            f"Generated embeddings for `{data.get('title', '')}` "
                            f"({len(data.get('text', '').split()):,} words).\n"
                        )
                st.session_state["scholar_ai"] = ScholarDigestAI()

            # Data loaded successfully
            st.session_state["dataset_loaded"] = True

    # Main panel for asking questions
    st.subheader("Ask Questions About the Papers")
    question_input = st.text_area("Your Question:")
    techincal_level = st.selectbox(
        "Select Technical Level",
        ["elementary", "high school", "undergrad", "domain expert"],
        index=3,
    )
    model_choice = st.selectbox(
        "Select Model",
        [
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/ministral-8b",
            "google/gemini-flash-1.5",
            "deepseek/deepseek-chat",
        ],
        index=2,
    )
    ask_btn = st.button("Ask AI")

    if ask_btn:
        if not st.session_state["dataset_loaded"]:
            st.error("Please load data before asking questions.")
            return

        if not question_input.strip():
            st.error("Please enter a question.")
            return

        # Access DOIs from session state
        n_dois = len(st.session_state["doi_list"])
        # st.write(f"n DOIs: {n_dois}")

        relevant_sections = None
        if n_dois > 1:
            # multiple DOIs => search embeddings db
            relevant_sections = search_database(
                query=question_input,
                model=embedding_model,
                db=embeddings_db,
                top_k=min(embeddings_db.n_docs, 20),
            )
            # st.markdown("### Relevant Sections:")
            # for section in relevant_sections:
            #     st.write("-"*50)
            #     st.write(section)
        # elif n_dois == 1:
        #     st.success("Giving LLM whole paper.\n")
        # else:
        #     st.warning("No papers loaded.\n")

        # Ask the LLM
        with st.spinner(f"{emoji.emojize(':robot:')} AI is thinking..."):
            response = st.session_state["scholar_ai"].ask_question(
                question=question_input,
                technical_level=techincal_level,
                model=model_choice,
                relevant_sections=relevant_sections,
            )

        st.markdown("### AI Response:")
        st.markdown(response)


if __name__ == "__main__":
    main()

import streamlit as st
from dotenv import load_dotenv

from src.loaders import load_documents_from_folder
from src.chunking import split_documents
from src.embeddings import create_vector_store
from src.retriever import get_retriever
from src.qa_chain import build_qa_chain

# Load environment variables from .env
load_dotenv()

# Streamlit page settings
st.set_page_config(page_title="Document Q&A Assistant", layout="wide")

# App title and description
st.title("📄 Document Q&A Assistant")
st.markdown(
    """
    Ask questions about documents stored in the `data/raw` folder.

    **How it works:**
    1. Put PDF or TXT files into `data/raw`
    2. Click **Process Documents**
    3. Ask a question in plain English
    """
)

# Session state to keep the QA chain available
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Button to process documents
if st.button("Process Documents"):
    with st.spinner("Loading, chunking and indexing documents..."):
        try:
            docs = load_documents_from_folder("data/raw")

            if not docs:
                st.warning("No PDF or TXT documents were found in data/raw.")
            else:
                chunks = split_documents(docs)
                vector_store = create_vector_store(chunks)
                retriever = get_retriever(vector_store)
                qa_chain = build_qa_chain(retriever)

                st.session_state.qa_chain = qa_chain
                st.session_state.documents_processed = True

                st.success(f"Processed {len(docs)} document sections into {len(chunks)} chunks.")

        except Exception as e:
            st.error(f"An error occurred while processing documents: {e}")

# Question input
query = st.text_input("Ask a question about your documents:")

# Run query
if query:
    if not st.session_state.documents_processed or st.session_state.qa_chain is None:
        st.warning("Please click 'Process Documents' first.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = st.session_state.qa_chain({"query": query})

                st.subheader("Answer")
                st.write(response["result"])

                st.subheader("Source Chunks")
                source_docs = response.get("source_documents", [])

                if source_docs:
                    for i, doc in enumerate(source_docs, start=1):
                        with st.expander(f"Source {i}"):
                            st.write(doc.page_content)

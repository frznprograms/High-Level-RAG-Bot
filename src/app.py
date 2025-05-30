import streamlit as st
from generator import Generator
from data_loader import Dataloader
from retriever import Retriever

st.set_page_config(page_title="RAG Demo", layout="centered")

st.subheader("Hi, I'm")
st.title("Hammond")

# start new instance of model for new user
if "rag_generator" not in st.session_state:
    loader = Dataloader()
    documents = loader.load_documents()
    splits = loader.chunk_documents()
    retriever = Retriever()
    retriever.initialize(splits)
    new_generator = Generator(retriever=retriever)
    new_generator.init_rag_chain()

    st.session_state.rag_generator = new_generator


query = st.text_input("Ask me anything!")
if query:
    with st.spinner("Generating answer..."):
        answer = st.session_state.rag_generator.query(query)
        st.markdown("### Answer:")
        st.success(answer)
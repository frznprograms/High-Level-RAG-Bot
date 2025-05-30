import os
import yaml
import streamlit as st
from generator import Generator
from data_loader import Dataloader
from retriever import Retriever
from bs4 import BeautifulSoup

st.set_page_config(page_title="Hammond", layout="centered")
st.title("Hi, I'm Hammond.")
st.markdown("Ask me anything, and I'll retrieve and respond with knowledge!")

# Instantiate RAG generator
if "rag_generator" not in st.session_state:
    loader = Dataloader()
    documents = loader.load_documents()
    splits = loader.chunk_documents()
    retriever = Retriever()
    retriever.initialize(splits)
    generator = Generator(retriever)
    generator.init_rag_chain()
    st.session_state.rag_generator = generator

# Chat history log (user-facing)
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Optional: sanitize model output to remove raw HTML
def clean_response(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator="\n")
    except Exception:
        return text

# Handle submission from input box
def handle_submit():
    query = st.session_state.user_input
    if not query.strip():
        return
    response = st.session_state.rag_generator.query(query)
    clean_resp = clean_response(response)
    st.session_state.chat_log.append(("You", query))
    st.session_state.chat_log.append(("Hammond", clean_resp))
    st.session_state.user_input = ""  # Clear input cleanly

# User input field
st.text_input("Ask me anything!", key="user_input", on_change=handle_submit)

# Scrollable chat area
st.markdown("### 🗨️ Conversation")

base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "../config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
chat_html = config.get("css", {}).get("design", "")

for sender, message in st.session_state.chat_log:
    role_class = "user-message" if sender == "You" else "ai-message"
    chat_html += f'<div class="chat-message {role_class}"><b>{sender}:</b><br>{message}</div>'

chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

st.divider()
# Reset button
if st.button("🔄 Reset Session"):
    for key in ["rag_generator", "chat_log", "user_input"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()
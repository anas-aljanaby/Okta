import yaml
from models import RAGHandler
import streamlit as st

VERBOSE = 1
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def initialize_handler():
    return RAGHandler(verbose=VERBOSE, add_context=True)


if "handler" not in st.session_state:
    rag_handler = RAGHandler(verbose=VERBOSE, add_context=True)
    st.session_state["handler"] = rag_handler

st.title("Okta AI - chat with local files")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.session_state["handler"] = initialize_handler()
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_files = st.sidebar.file_uploader(
    "Upload", label_visibility="collapsed", accept_multiple_files=True
)

for file in uploaded_files:
    st.session_state["handler"].retriever.save_embedding(file)

prompt = st.chat_input("say something")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = st.session_state.handler.make_prediction(st.session_state.messages)
    with st.chat_message("assistant"):
        st.write(response["content"])
    st.session_state.messages.append(response)

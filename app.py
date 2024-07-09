import yaml
from rag_handler import RAGHandler
import streamlit as st
from utils import create_word_document


def offer_download(message_content, i):
    word_file = create_word_document(message_content)
    st.download_button(
        label="💾",
        data=word_file,
        file_name="message.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"download_{i}",
    )


VERBOSE = 1
DEBUG = 0
llm_type = "hf"
llm_name = "default"
embeddings_model = "bert-base-multilingual-cased"


def initialize_handler(model_type="hf", model_name="default"):
    return RAGHandler(
        model_type=model_type,
        model_name=model_name,
        add_system_prompt=False,
        embeddings_model_name=embeddings_model,
        verbose=VERBOSE,
        add_context=True,
    )


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


if "files" not in st.session_state:
    st.session_state["files"] = []

if "handler" not in st.session_state:
    st.session_state["handler"] = initialize_handler(llm_type, llm_name)

st.title("Okta AI - chat with local files")

if "messages" not in st.session_state:
    if DEBUG:
        st.session_state.messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": 'Answer:\nThe answer to your question is not in the provided context as no context has been provided. Please provide more information or clarify your question so I can assist you better. Without context, it is impossible to determine the meaning of "asd."',
            },
        ]
    else:
        st.session_state.messages = []

if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.session_state["handler"] = initialize_handler(llm_type, llm_name)
    st.rerun()

st.markdown(
    """
    <style>
    .st-emotion-cache-15hul6a.ef3psqc12 {
        font-size: 15px;
    }
    .st-emotion-cache-fm8pe0.e1nzilvr4 p {
        font-size: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

uploaded_files = st.sidebar.file_uploader(
    "Upload", label_visibility="collapsed", accept_multiple_files=True
)

for file in uploaded_files:
    if file.name in st.session_state.files:
        if VERBOSE:
            print(file.name, "already loaded")
        continue
    print(file.type)
    st.session_state.files.append(file.name)
    st.session_state.handler.retriever.save_embedding(file)

prompt = st.chat_input("say something")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = st.session_state.handler.make_prediction(st.session_state.messages)
    st.session_state.messages.append(response)

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            cols = st.columns([0.06, 0.06, 0.88])
            with cols[0]:
                offer_download(message["content"], i)
            with cols[1]:
                if st.button("↻", key=f"reg_{i}"):
                    pass

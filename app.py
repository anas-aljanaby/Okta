import yaml
from rag_handler import RAGHandler
import streamlit as st
from utils import create_word_document
import uuid
import json
import os


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
llm_type = "openai"
llm_name = "default"
# embeddings_model = "bert-base-multilingual-cased"
embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"


def initialize_handler(model_type="hf", model_name="default"):
    return RAGHandler(
        model_type=model_type,
        model_name=model_name,
        add_system_prompt=False,  # TODO: add system prompt
        embeddings_model_name=embeddings_model,
        verbose=VERBOSE,
        add_context=True,
    )


def create_new_chat():
    chat_id = str(uuid.uuid4())
    new_chat = {
        "chat_id": chat_id,
        "messages": [],
    }
    with open(f"chats/{chat_id}.json", "w") as file:
        json.dump(new_chat, file)
    return chat_id


def load_chat(chat_id):
    with open(f"chats/{chat_id}.json", "r") as file:
        chat = json.load(file)
    return chat


def save_message(chat_id, role, content):
    chat_data = load_chat(chat_id)
    new_message = {
        "role": role,
        "content": content,
    }
    chat_data["messages"].append(new_message)
    with open(f"chats/{chat_id}.json", "w") as chat_file:
        json.dump(chat_data, chat_file)


if not os.path.exists("chats"):
    os.mkdir("chats")

if "welcome" not in os.listdir("chats"):
    welcome_chat = {
        "chat_id": "welcome",
        "messages": [
            {
                "role": "user",
                "content": "Welcome to the Okta AI chat interface. Please select a chat or create a new one to get started.",
            },
            {
                "role": "assistant",
                "content": "Messages by the LLM will be displayed like this. You can download the message as a Word document by clicking the disk icon.",
            },
        ],
    }
    with open(f"chats/welcome.json", "w") as file:
        json.dump(welcome_chat, file)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

if "files" not in st.session_state:
    st.session_state["files"] = []

if "handler" not in st.session_state:
    st.session_state["handler"] = initialize_handler(llm_type, llm_name)

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
        st.session_state.messages = load_chat("welcome")
        st.session_state.current_chat = "welcome"


def rename_chat():
    current_chat = st.session_state.get("current_chat", "welcome")
    new_name = st.text_input(
        "Rename current chat", value=current_chat, key="new_chat_name"
    )
    if st.button("Confirm Rename", key="confirm_rename_button"):
        st.write("Button clicked!")
        if new_name and new_name != current_chat:
            new_name = new_name.strip()
            new_name = new_name.replace(" ", "_")
            try:
                old_path = f"chats/{current_chat}.json"
                new_path = f"chats/{new_name}.json"
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    st.session_state.current_chat = new_name
                    st.success(f"Chat renamed to: {new_name}")
                    st.experimental_rerun()
                else:
                    st.error(f"Chat file {old_path} not found.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        elif new_name == current_chat:
            st.info("The new name is the same as the current name. No changes made.")
        else:
            st.warning("Please enter a valid name.")


def delete_chat():
    if st.button("Delete Chat"):
        chat_id = st.session_state.current_chat
        if os.path.exists(f"chats/{chat_id}.json"):
            os.remove(f"chats/{chat_id}.json")
            st.session_state.current_chat = None
            st.rerun()

cols = st.columns([0.15, 0.2, 0.2, 0.2, 0.2])
if "settings_open" not in st.session_state:
    st.session_state.settings_open = False
with cols[0]:
    if st.button("Chat Settings"):
        st.session_state.settings_open = not st.session_state.settings_open
if st.session_state.settings_open:
    with cols[1]:
        rename_chat()
    with cols[2]:
        delete_chat()

st.title("Okta AI - chat with local files")

retrieval_type = st.sidebar.selectbox(
    "Choose Retrieval Type", ("Snippets from Document", "Full Document")
)

if retrieval_type == "Full Document":
    st.session_state.handler.mode = "full"
else:
    st.session_state.handler.mode = "snippets"

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
    st.session_state.files.append(file.name)
    st.session_state.handler.retriever.save_embedding(file)


if st.sidebar.button("New Chat"):
    st.session_state.current_chat = create_new_chat()
    st.session_state.messages = load_chat(st.session_state.current_chat)
    st.session_state["handler"] = initialize_handler(llm_type, llm_name)

button_style = """
    <style>
    .custom-chat-button-container button {
        background-color: transparent;
        color: #fff;
        padding: 1px 1px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-bottom: 0px;
        width: 100%;
    }
    .custom-chat-button-container button:hover {
        background-color: #2b2c36;
        color: #fff;
    }
    </style>
"""
st.sidebar.markdown(button_style, unsafe_allow_html=True)

st.markdown(
    """
<style>.element-container:has(#button-after) + div button {
        background-color: transparent;
        color: #fff;
        padding: 1px 1px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-bottom: 0px;
        width: 100%;
        }
        .element-container:has(#button-after) + div button:hover {
        background-color: #40404e;
        color: #fff;
        }
 </style>""",
    unsafe_allow_html=True,
)

chats = [chat.split(".")[0] for chat in os.listdir("chats") if chat.endswith(".json")]
print(chats)
for chat in chats:
    st.sidebar.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    if st.sidebar.button(chat):
        st.session_state.current_chat = chat
        st.session_state.messages = load_chat(chat)
        chat_selected = True


prompt = st.chat_input("say something")
if prompt:
    print(st.session_state.messages)
    st.session_state.messages["messages"].append({"role": "user", "content": prompt})
    save_message(st.session_state.current_chat, "user", prompt)
    response = st.session_state.handler.make_prediction(
        st.session_state.messages["messages"]
    )
    save_message(st.session_state.current_chat, "assistant", response["content"])
    st.session_state.messages["messages"].append(response)


chat_id, messages = (
    st.session_state.messages["chat_id"],
    st.session_state.messages["messages"],
)

for i, msg in enumerate(messages):
    role, content = msg["role"], msg["content"]
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            cols = st.columns([0.06, 0.06, 0.88])
            with cols[0]:
                offer_download(content, i)
            with cols[1]:
                if st.button("↻", key=f"reg_{i}"):
                    pass

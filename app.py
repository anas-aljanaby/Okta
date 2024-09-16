import yaml
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from utils import create_word_document
import uuid
import json
import os

from session_managers import SessionRetriever

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
        background-color: #33353f;
        color: #fff;
        }
 </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>.element-container:has(#button-after1) + div button {
        background-color: #33353f;
        color: #fff;
        padding: 1px 1px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-bottom: 0px;
        width: 100%;
        }
        .element-container:has(#button-after1) + div button:hover {
        background-color: #33353f;
        color: #fff;
        }
 </style>""",
    unsafe_allow_html=True,
)


def create_new_chat():
    chat_id = str(uuid.uuid4())
    new_chat = {
        "chat_id": chat_id,
        "display_name": chat_id,
        "uploaded_files": [],
        "messages": [],
    }
    chat_path = f"chats/{chat_id}.json"
    with open(chat_path, "w") as file:
        json.dump(new_chat, file)
    metadata_path = "chats/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as file:
            metadata = json.load(file)
    else:
        metadata = {"chat_ids": []}
    metadata["chat_ids"].insert(0, chat_id)  # Add new chat ID to the top
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)
    return chat_id


def load_chat(chat_id):
    with open(f"chats/{chat_id}.json", "r") as file:
        chat = json.load(file)
    return chat


def load_chats():
    metadata_path = "chats/metadata.json"
    if not os.path.exists(metadata_path):
        return []

    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    chat_ids = metadata.get("chat_ids", [])
    chats = []
    for chat_id in chat_ids:
        chat_path = f"chats/{chat_id}.json"
        if os.path.exists(chat_path):
            with open(chat_path, "r") as file:
                chat_data = json.load(file)
                chats.append(chat_data)
    return chats


def load_chat_session(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.chat = load_chat(st.session_state.current_chat_id)
    st.session_state.messages = st.session_state.chat["messages"]
    st.session_state.uploaded_files = st.session_state.chat["uploaded_files"]
    st.session_state.session_retriever = SessionRetriever(
        st.session_state.current_chat_id
    )
    st.session_state.retrieve_tool = create_retriever_tool(
        st.session_state.session_retriever.as_retriever(),
        "file_retriever",
        "Searches and return chunks from the embedded files.",
    )
    st.session_state.agent_executer = create_react_agent(
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        [st.session_state.retrieve_tool],
    )
    st.rerun()


def save_message(chat_id, role, content):
    chat_data = load_chat(chat_id)
    new_message = {
        "role": role,
        "content": content,
    }
    chat_data["messages"].append(new_message)
    with open(f"chats/{chat_id}.json", "w") as chat_file:
        json.dump(chat_data, chat_file)


def add_uploaded_file_name(filename):
    chat_data = load_chat(st.session_state.current_chat_id)
    chat_data["uploaded_files"].append(filename)
    with open(f"chats/{st.session_state.current_chat_id}.json", "w") as chat_file:
        json.dump(chat_data, chat_file)


def delete_chat():
    chat_id = st.session_state.current_chat_id
    chat_path = f"chats/{chat_id}.json"
    os.remove(chat_path)
    metadata_path = "chats/metadata.json"
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    metadata["chat_ids"].remove(chat_id)
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)
    if metadata["chat_ids"]:
        load_chat_session(metadata["chat_ids"][0])
    else:
        load_chat_session(create_new_chat())


if not os.path.exists("chats"):
    os.mkdir("chats")

VERBOSE = 1
DEBUG = 0


def rename_chat():
    current_chat = st.session_state.chat["display_name"]
    new_name = st.text_input(
        "Enter new chat name", value=current_chat, key="new_chat_name"
    )
    if st.button("Confirm Rename", key="confirm_rename_button"):
        if new_name and new_name != current_chat:
            st.session_state.chat["display_name"] = new_name
            chat_path = f"chats/{st.session_state.current_chat_id}.json"
            with open(chat_path, "w") as file:
                json.dump(st.session_state.chat, file)
            st.success(f"Chat renamed to: {new_name}")
            st.rerun()


if "settings_open" not in st.session_state:
    st.session_state.settings_open = False

if "rename_button_open" not in st.session_state:
    st.session_state.rename_button_open = False

if st.button("Chat Settings"):
    st.session_state.settings_open = not st.session_state.settings_open

cols = st.columns([0.5, 0.5])


if "current_chat_id" not in st.session_state:
    if load_chats():
        load_chat_session(load_chats()[0]["chat_id"])
    else:
        load_chat_session(create_new_chat())


uploaded_files = st.sidebar.file_uploader(
    "Upload",
    label_visibility="collapsed",
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.current_chat_id}",
)

for file in uploaded_files:
    if file.name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(file.name)
        st.session_state.session_retriever.add_document(file)
        add_uploaded_file_name(file.name)

with cols[0]:
    if st.session_state.settings_open:
        if st.button("Delete current Chat"):
            delete_chat()
        if st.button("Rename current Chat"):
            st.session_state.rename_button_open = (
                not st.session_state.rename_button_open
            )
        if st.session_state.rename_button_open:
            rename_chat()
with cols[1]:
    if st.session_state.chat["uploaded_files"]:
        st.write("**Embedded Files:**")
        for file in st.session_state.chat["uploaded_files"]:
            st.write(f"- {file}")

st.title("Okta AI - chat with local files")

if st.sidebar.button("New Chat"):
    load_chat_session(create_new_chat())


for chat in load_chats():
    if chat["chat_id"] == st.session_state.current_chat_id:
        st.sidebar.markdown('<span id="button-after1"></span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    if st.sidebar.button(chat["display_name"]):
        load_chat_session(chat["chat_id"])


def extract_llm_response(response):
    messages = response["messages"]
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content


prompt = st.chat_input("say something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.current_chat_id, "user", prompt)
    response = st.session_state.agent_executer.invoke(
        {"messages": [HumanMessage(prompt)]}
    )
    # print(response)
    content = extract_llm_response(response)
    save_message(st.session_state.current_chat_id, "assistant", content)
    st.session_state.messages.append({"role": "assistant", "content": content})

for i, msg in enumerate(st.session_state.messages):
    role, content = msg["role"], msg["content"]
    with st.chat_message(role):
        st.markdown(content)

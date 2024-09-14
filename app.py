import yaml
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser
from utils import create_word_document
import uuid
import json
import os

from session_managers import SessionRetriever

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

def offer_download(message_content, i):
    word_file = create_word_document(message_content)
    st.download_button(
        label="💾",
        data=word_file,
        file_name="message.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"download_{i}",
    )

def create_new_chat():
    chat_id = str(uuid.uuid4())
    new_chat = {
        "chat_id": chat_id,
        "uploaded_files": [],
        "messages": [],
    }
    with open(f"chats/{chat_id}.json", "w") as file:
        json.dump(new_chat, file)
    return chat_id


def load_chat(chat_id):
    with open(f"chats/{chat_id}.json", "r") as file:
        chat = json.load(file)
    return chat

def load_chat_session(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.chat = load_chat(st.session_state.current_chat_id)
    st.session_state.messages = st.session_state.chat["messages"]
    st.session_state.uploaded_files = st.session_state.chat["uploaded_files"]
    st.session_state.session_retriever = SessionRetriever(st.session_state.current_chat_id)
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

if not os.path.exists("chats"):
    os.mkdir("chats")

VERBOSE = 1
DEBUG = 0


# if 'chat' not in st.session_state:
#     st.session_state.current_chat_id = create_new_chat()
#     st.session_state.chat = load_chat(st.session_state.current_chat_id)
#     st.session_state.messages = st.session_state.chat["messages"]
#
# if "session_retriever" not in st.session_state:
#     st.session_state.session_retriever = SessionRetriever(st.session_state.current_chat_id)
#
# if "retrieve_tool" not in st.session_state:
#     st.session_state.retrieve_tool = create_retriever_tool(
#         st.session_state.session_retriever.as_retriever(),
#         "file_retriever",
#         "Searches and return chunks from the embedded files.",
#     )
#
# if 'parser' not in st.session_state:
#     st.session_state.parser = StrOutputParser()
#
# if "agent_executer" not in st.session_state:
#     st.session_state.agent_executer = create_react_agent(
#         ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
#         [st.session_state.retrieve_tool],
#     )
#

cols = st.columns([0.15, 0.2, 0.2, 0.2, 0.2])
if "settings_open" not in st.session_state:
    st.session_state.settings_open = False
with cols[0]:
    if st.button("Chat Settings"):
        st.session_state.settings_open = not st.session_state.settings_open
if st.session_state.settings_open:
    with cols[1]:
        pass
    with cols[2]:
        pass

st.title("Okta AI - chat with local files")

if "current_chat_id" not in st.session_state:
    load_chat_session(create_new_chat())

uploaded_files = st.sidebar.file_uploader(
    "Upload", label_visibility="collapsed", accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.current_chat_id}")

for file in uploaded_files:
    if file.name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(file.name)
        st.session_state.session_retriever.add_document(file)
        add_uploaded_file_name(file.name)


if st.sidebar.button("New Chat"):
    load_chat_session(create_new_chat())

for chat in os.listdir("chats"):
    st.sidebar.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    if st.sidebar.button(chat):
        load_chat_session(chat.split(".")[0])

def extract_llm_response(response):
    messages = response['messages']
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
        if role == "assistant":
            cols = st.columns([0.06, 0.06, 0.88])
            with cols[0]:
                offer_download(content, i)
            with cols[1]:
                if st.button("↻", key=f"reg_{i}"):
                    pass


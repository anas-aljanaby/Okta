![Build Status](https://github.com/anas-aljanaby/Okta/actions/workflows/python-ci.yml/badge.svg)
# Okta AI
Okta is a RAG (Retrieval-Augmented Generation) App that enables users to run language models (LLMs) on local files, it helps organize your file directories into seperate chat sessions.
![Description of the Screenshot](./images/chat-screenshot.png)
## Quick Start
1. Clone the repository
2. Install the required packages
3. Run the app
```bash
git clone https://github.com/anas-aljanaby/Okta.git
cd Okta
pip install -r requirements.txt
streamlit run app.py
```
## Usage
Starting a New Chat
1. Create a New Chat:
- Begin by clicking on the New Chat button located in the app's main interface.
2. Upload Files:
- Once you've started a new chat, you can upload a file or multiple files directly from the sidebar. These files will serve as the knowledge base for the language model during this chat session.
3. Drag and Drop Directories:
- If you prefer, you can also drag and drop an entire directory into the sidebar. The app will automatically organize and index the contents of the directory, making all of its files accessible to the LLM.
4. Ask Questions:
- With your files uploaded, the LLM will use this specific set of documents to answer any questions you pose during the chat session. This ensures that responses are contextually relevant and based on the information you've provided.
## Contributing
Any contributions are welcome.
1. Fork the repository
2. Create a new branch (`git checkout -b feature/feature-name`)
3. Commit your changes (`git commit -am 'Add a feature'`)
4. Push to the branch (`git push origin feature/feature-name`)
5. Open a new Pull Request

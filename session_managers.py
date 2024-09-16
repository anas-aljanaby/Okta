from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

from file_parser import parse_file


class SessionRetriever:
    def __init__(self, session_id, embedding_model=None):
        self.session_id = session_id
        self.persist_directory = "./persist_directory"
        self.embedding_model = (
            embedding_model or OpenAIEmbeddings()
        )  # Default to OpenAI Embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.chroma_store = Chroma(
            collection_name=f"chroma_{self.session_id}",
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def add_document(self, document):
        text = parse_file(document)
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        self.chroma_store.add_documents(documents)

    def retrieve(self, query, top_k=3):
        results = self.chroma_store.similarity_search(query, top_k=top_k)
        return results

    def as_retriever(self):
        return self.chroma_store.as_retriever()

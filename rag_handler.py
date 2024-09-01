from models import LLModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
import shutil
import yaml

from file_parser import parse_file


class Retriever:
    def __init__(
        self,
        embeddings_model,
        embeddings_folder="embeddings",
        chunk_size=300,
        top_k=3,
        verbose=0,
    ):
        self.embeddings_model_name = embeddings_model
        self.tokenizer = AutoTokenizer.from_pretrained(embeddings_model)
        self.embeddings_model = AutoModel.from_pretrained(embeddings_model)
        self.embeddings_folder = embeddings_folder
        self.chunk_size = chunk_size
        self.embedded_files = dict()
        self.verbose = verbose
        self.top_k = top_k
        self.context = ""
        self.clear_embeddings()

    def get_embeddings(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.embeddings_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.detach().numpy()

    def chunk_text(self, text):
        words = text.split()
        chunks = [
            " ".join(words[i : i + self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]
        return chunks

    def clear_embeddings(self):
        if os.path.exists(self.embeddings_folder):
            shutil.rmtree(self.embeddings_folder)
        os.makedirs(self.embeddings_folder)
        file = open(os.path.join(self.embeddings_folder, "index.json"), "w")
        file.close()

    def save_embedding(self, file):
        text = parse_file(file)
        chunks = self.chunk_text(text)
        for idx, chunk in enumerate(chunks):
            embedding = self.get_embeddings(chunk)
            embedding_file = os.path.join(
                self.embeddings_folder, f"{file.name}_chunk_{idx}.npy"
            )

            np.save(embedding_file, embedding)
            with open(
                os.path.join(self.embeddings_folder, "index.json"), "a"
            ) as index_file:
                index_entry = {
                    "file_name": file.name,
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "embedding_file": embedding_file,
                }
                json.dump(index_entry, index_file)
                index_file.write("\n")
                if self.verbose:
                    print(f"wrote embedding for {file.name} at {embedding_file}")
        self.embedded_files[file.name] = text

    def load_embeddings(self):
        embeddings = []
        metadata = []
        index_file_path = os.path.join(self.embeddings_folder, "index.json")

        if os.path.exists(index_file_path):
            with open(index_file_path, "r") as index_file:
                for line in index_file:
                    entry = json.loads(line.strip())
                    embedding = np.load(entry["embedding_file"])
                    embeddings.append(embedding)
                    metadata.append(entry)
        return np.vstack(embeddings), metadata

    def get_most_similar(self, query, top_k=5):
        query_embedding = self.get_embeddings(query)
        embeddings, metadata = self.load_embeddings()
        similarities = cosine_similarity(query_embedding, embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        if self.verbose:
            for idx in top_indices:
                print(metadata[idx]["file_name"])
                print(metadata[idx]["chunk_index"])
                print(metadata[idx]["chunk_text"])
                print(similarities[idx])
        return [metadata[idx]["chunk_text"] for idx in top_indices]


class RAGHandler:
    def __init__(
        self,
        model_type="hf",
        model_name="default",
        embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2",
        add_context=False,
        add_system_prompt=True,
        top_k=3,
        verbose=0,
        mode="snippets",
    ):
        self.model = LLModel(
            model_type=model_type,
            model_name=model_name,
            add_system_prompt=add_system_prompt,
        )
        self.retriever = Retriever(
            embeddings_model=embeddings_model_name,
            top_k=top_k,
            verbose=verbose,
        )
        self.add_context = add_context
        self.top_k = top_k
        self.mode = mode
        self.verbose = verbose
        self.system_message = self.load_system_message()

    def load_system_message(self):
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config["model"]["system_prompt"]

    def retrieve_files(self):
        context = ""
        for file_name in self.retriever.embedded_files:
            context += f"{file_name}\n{self.retriever.embedded_files[file_name]}"
        return context

    def retrieve_context(self, message_content=None):
        if self.mode == "snippets":
            context = self.retriever.get_most_similar(message_content, top_k=self.top_k)
            context = "\n".join(context)
        else:
            context = self.retrieve_files()
        return context

    def trim_history(self):
        CONTEXT_LENGTH_LIMIT = 10000  # Example value, adjust as necessary
        current_length = sum(
            len(message["content"].split())
            for message in self.model.language_model.history
        )
        print(current_length)
        while (
            current_length > CONTEXT_LENGTH_LIMIT
            and len(self.model.language_model.history) > 1
        ):
            removed_message = self.model.language_model.history.pop(0)
            current_length -= len(removed_message["content"])

    def make_prediction(self, messages):
        if not isinstance(messages, list):
            raise ValueError(f"messages must be a list of dicts, got {type(messages)}")
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Each item in messages must be a dict, got {type(item)}"
                )
            if "content" not in item:
                raise ValueError(f"Each item in messages must have a 'content' key")
            if "role" not in item:
                raise ValueError(f"Each item in messages must have a 'role' key")
        if self.retriever.embedded_files:
            context = self.retrieve_context(messages[-1]["content"])
        else:
            context = ""
        msg = messages[-1]["content"] + "\nContext:" + context
        user_message = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": msg},
        ]
        if self.verbose:
            print(user_message)
        prediction = self.model.language_model.predict(user_message)
        return prediction

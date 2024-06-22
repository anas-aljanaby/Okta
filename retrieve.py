from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
import shutil


class RAGHandler:
    def __init__(
        self,
        embeddings_folder='embeddings',
        chunk_size=150, 
        verbose=0,
                 ):
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.embeddings_folder = embeddings_folder
        self.chunk_size = chunk_size
        self.verbose = verbose
        os.makedirs(self.embeddings_folder, exist_ok=True)

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.detach().numpy()

    def save_file_ebmeddings(self, file_path):
        base_name = os.path.basename(file_path)
        embeddings_file = os.path.join(self.embeddings_folder, f'{base_name}.npy')
        with open(file_path, 'r') as file:
            text = file.read()
        text_parts = self.chunk_text(text)
        text_embeddings = np.vstack([self.get_embeddings(part) for part in text_parts])
        np.save(embeddings_file, text_embeddings)
        return text_parts, text_embeddings

    def chunk_text(self, text):
        words = text.split()
        chunks = [' '.join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
        return chunks 

    def clear_embeddings(self):
        if os.path.exists(self.embeddings_folder):
            shutil.rmtree(self.embeddings_folder)
        os.makedirs(self.embeddings_folder)
        file = open(os.path.join(self.embeddings_folder, 'index.json'), 'w')
        file.close()

    def save_embedding(self, file_path, overwrite=False):
        if self.verbose:
            print('started saving')
        if not overwrite:
            with open(os.path.join(self.embeddings_folder, 'index.json'), 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry['file_path'] == file_path:
                        print(f'{os.path.basename(file_path)} already loaded')
                        return 
        with open(file_path, 'r') as file:
            text = file.read()
        chunks = self.chunk_text(text)
        for idx, chunk in enumerate(chunks):
            embedding = self.get_embeddings(chunk)
            embedding_file = os.path.join(self.embeddings_folder, f"{os.path.basename(file_path)}_chunk_{idx}.npy")

            np.save(embedding_file, embedding)
            with open(os.path.join(self.embeddings_folder, 'index.json'), 'a') as index_file:
                index_entry = {
                    "file_path": file_path,
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "embedding_file": embedding_file
                }
                json.dump(index_entry, index_file)
                index_file.write("\n")
                if self.verbose:
                    print(f'wrote embedding for {os.path.basename(file_path)} at {embedding_file}')

    def load_embeddings(self):
        embeddings = []
        metadata = []
        index_file_path = os.path.join(self.embeddings_folder, 'index.json')

        if os.path.exists(index_file_path):
            with open(index_file_path, 'r') as index_file:
                for line in index_file:
                    entry = json.loads(line.strip())
                    embedding = np.load(entry['embedding_file'])
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
                print(metadata[idx]['file_path'])
                print(metadata[idx]['chunk_index'])
                print(metadata[idx]['chunk_text'])
                print(similarities[idx])
        return [metadata[idx]['chunk_text'] for idx in top_indices]

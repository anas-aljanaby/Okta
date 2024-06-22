import gradio as gr
from huggingface_hub import InferenceClient
import os
import yaml
from retrieve import RAGHandler

VERBOSE = 0 
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta",
                         token=os.getenv('HF_API_TOKEN'))
rag_handler = RAGHandler(verbose=VERBOSE)
rag_handler.clear_embeddings()

system_prompt =  {'role': 'system',
     'content': config['model']['system_prompt']
    }

context = 'None'
hf_history = [system_prompt]
uploaded_files = []

def get_current_chat(message, gr_history):
    global context, hf_history
    round = []
    if gr_history:
        _, ai = gr_history[-1]
        round.append({'role': 'assistant', 'content': ai})
    else:
        hf_history = [system_prompt]
    if len(uploaded_files):
        context = '\n'.join(rag_handler.get_most_similar(message, top_k=3))
    text = f'Context:\n{context}\nPrompt:\n{message}'
    round.append({'role': 'user', 'content': text})
    return round


def chat(message, history):
    global hf_history
    hf_history += get_current_chat(message, history)
    client_resp = client.chat_completion(hf_history, max_tokens=496)
    content = client_resp.choices[0].message.content
    return content

uploaded_files = []
def display_files(files):
    global uploaded_files
    if files:
        for file in files:
            rag_handler.save_embedding(file)
            uploaded_files.extend([os.path.basename(file) for file in files])
    return '\n'.join(uploaded_files)

css = """
#chat-container {
    height: calc(100vh - 8em); /* Adjust this if you have a header or footer */
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            upload_button = gr.UploadButton(file_count="multiple")
            file_display = gr.Textbox(label="Ingested Files", interactive=False)
        with gr.Column(scale=4, elem_id="chat-container"):
            chat_ui = gr.ChatInterface(chat)
    upload_button.upload(display_files, inputs=upload_button, outputs=file_display)

demo.launch()

from openai import OpenAI
from huggingface_hub import InferenceClient
import yaml
import os

DEFAULT_HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"


class Model:
    def __init__(self, model_name, max_tokens=496, add_system_prompt=True):
        self.model_name = model_name
        self.max_tokens = max_tokens
        if add_system_prompt:
            with open("config.yaml", "r") as file:
                config = yaml.safe_load(file)
            self.history = [
                {"role": "system", "content": config["model"]["system_prompt"]}
            ]
        else:
            self.history = []

    def predict(self, message_history):
        return message_history


class HFModel(Model):
    def __init__(self, model_name="default", add_system_prompt=True):
        super().__init__(model_name, add_system_prompt=add_system_prompt)
        if model_name == "default":
            self.client = InferenceClient(
                model=DEFAULT_HF_MODEL, token=os.getenv("HF_API_TOKEN")
            )
        else:  # TODO: ADD ERROR HANDLING
            self.client = InferenceClient(
                model=model_name, token=os.getenv("HF_API_TOKEN")
            )

    def predict(self, message_history):
        client_resp = self.client.chat_completion(
            message_history, max_tokens=self.max_tokens
        )
        content = client_resp.choices[0].message.content
        return {"role": "assistant", "content": content}


class OpenAIModel(Model):
    def __init__(self, model_name="default", add_system_prompt=True):
        super().__init__(model_name, add_system_prompt=add_system_prompt)
        if model_name == "default":
            self.model_name = DEFAULT_OPENAI_MODEL
        else:  # TODO: ADD ERROR HANDLING
            self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def predict(self, message_history):
        client_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history,
        )
        resp_content = client_response.choices[0].message.content
        return {"role": "assistant", "content": resp_content}


class LLModel:
    def __init__(
        self,
        model_type="hf",
        model_name="default",
        add_system_prompt=True,
    ):
        if model_type == "hf":
            self.language_model = HFModel(
                model_name,
                add_system_prompt=add_system_prompt,
            )
        elif model_type == "openai":
            self.language_model = OpenAIModel(
                model_name, add_system_prompt=add_system_prompt
            )
        else:
            raise ValueError(f"model of type {model_type} not supported")

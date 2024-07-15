import os

lmstudio_config_list = [
    {
        "model": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "api_type": "open_ai",
        "base_url": "http://localhost:1234/v1",
        "api_key": "sk-111111111111111111111111"
    }
]
gpt_config_list = [
    {
        "model": "gpt-3.5-turbo-16k",
        "api_key": os.environ.get("OPENAI_API_KEY")
    }
]
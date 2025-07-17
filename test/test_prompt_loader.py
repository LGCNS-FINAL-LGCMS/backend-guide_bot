import asyncio
import os

from app.utils.prompt_loader import load_prompt_from_yaml

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FILE_PATH = os.path.join(BASE_DIR, "prompts", "rag_prompt.yaml")

    text = load_prompt_from_yaml(DATA_FILE_PATH)

    print(text["system_template"])
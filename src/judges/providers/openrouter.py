import json
import requests
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.judges.providers.models import Model, OR_NAME_API
from src.judges.providers.base import APICall

class OpenRouterAPIRequest(APICall):
    def call(self, question: str, system: str) -> str:
        key_path = PROJECT_ROOT / "OpenRouterKEY.txt"
        try:
            with open(key_path, "r") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {key_path}. Please make sure OpenRouterKEY.txt exists.")

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": OR_NAME_API.get(self.model),
                "messages": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "temperature": 0.0
            }),
        )

        try:
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, ValueError) as e:
            print("Error parsing response:", response.text)
            raise e

def main():
    api_request = OpenRouterAPIRequest(Model.GEMMA)
    print("Calling OpenRouter API...")
    response_text = api_request.call("What is the meaning of life?", "You are a philosopher.")
    print("\nResponse:")
    print(response_text)

if __name__ == "__main__":
    main()
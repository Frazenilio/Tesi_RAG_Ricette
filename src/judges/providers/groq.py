import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.judges.providers import models as APIModels
from src.judges.providers.base import APICall
from groq import Groq

class GroqAPI(APICall):
    def call(self, question: str, system: str) -> str:
        key_path = PROJECT_ROOT / "GroqKEY.txt"
        try:
            with open(key_path, "r") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {key_path}. Please make sure GroqKEY.txt exists.")
            
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=APIModels.GROQ_NAME_API.get(self.model),
            messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": question
            }
            ],
            temperature=0,
            max_completion_tokens=8192,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content

def main():
    api_request = GroqAPI(APIModels.Model.QWEN)
    print("Calling Groq API...")
    response_text = api_request.call("What is the meaning of life?", "You are a philosopher.")
    print("\nResponse:")
    print(response_text)

if __name__ == "__main__":
    main()

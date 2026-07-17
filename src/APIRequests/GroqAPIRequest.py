import sys
from pathlib import Path

# Add project root to sys.path so we can import properly whether run directly or as a module
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.APIRequests import APIModels
from src.APIRequests.APICall import APICall
from groq import Groq

class GroqAPI(APICall):
    def call(self, question: str, system: str) -> str:
        key_path = Path(__file__).parent.parent.parent / "GroqKEY.txt"
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
            },
            ],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            # reasoning_effort="medium",
            stream=True,
            stop=None
        )

        full_response = ""
        for chunk in completion:
            full_response += chunk.choices[0].delta.content or ""
        
        return full_response
        
if __name__ == "__main__":
    api = GroqAPI(APIModels.Model.LLAMA)
    print(api.call("What is the square root of 4?", "You are an expert mathematician."))

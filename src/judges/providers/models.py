from enum import Enum

class Model(Enum):
    QWEN = "QWEN"
    GEMMA = "GEMMA"
    LLAMA = "LLAMA"
    MINISTRAL = "MINISTRAL"
    GPT = "GPT"

OR_NAME_API: dict[Model, str] = {
    Model.QWEN : "qwen/qwen3-next-80b-a3b-instruct:free",
    Model.GEMMA : "google/gemma-4-26b-a4b-it:free"
}

GROQ_NAME_API: dict[Model, str] = {
    Model.LLAMA : "llama-3.1-8b-instant",
    Model.GPT : "openai/gpt-oss-20b",
    Model.QWEN : "qwen/qwen3.6-27b"
}

from abc import ABC, abstractmethod
from src.judges.providers.models import Model

class APICall(ABC):
    def __init__(self, model: Model) -> None:
        self.model = model

    @abstractmethod
    def call(self, question: str, system: str) -> str:
        pass

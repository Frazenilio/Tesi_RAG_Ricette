from src.APIRequests import APIModels
from abc import ABC, abstractmethod

class APICall(ABC):
    def __init__(self, model: APIModels.Model) -> None:
        self.model = model

    @abstractmethod
    def call(self, question: str, system: str) -> str:
        pass

from abc import ABC, abstractmethod

class RetrievalModel(ABC):
    @abstractmethod
    def search(self, query_processed: str, top_n: int = 5):
        pass
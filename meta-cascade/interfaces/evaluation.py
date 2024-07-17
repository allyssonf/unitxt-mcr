from abc import ABC, abstractmethod
from typing import Any, Dict, List
from utils.files import save_json_file, save_pickle

class Evaluation(ABC):
    @abstractmethod
    def evaluate(self):
        pass

    def save_predictions(self, filename: str, data: Dict[str, Any] | List[Dict[str, Any]] | Any):
        save_pickle(filename, data)

    def save_results(self, filename: str, data: Dict[str, Any] | List[Dict[str, Any]] | Any):
        save_json_file(filename, data)
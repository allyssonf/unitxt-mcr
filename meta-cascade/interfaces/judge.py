from abc import ABC, abstractmethod


class Judge(ABC):

    @abstractmethod
    def infer(self):
        pass


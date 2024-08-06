import logging
import time
import datetime
import pickle
import os

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from utils.files import save_json_file, save_pickle, create_path, file_exists

logger = logging.getLogger(__name__)


class Evaluation(ABC):
    model_name: str = ""
    model_label: str = ""
    max_tokens: int = 32
    max_instances: int | None = None
    watsonx_model: bool = False
    bam_model: bool = False
    overwrite_results: bool = False
    results_folder: str = "results"
    predictions_folder: str = "predictions"

    def __init__(self, model_name, args):
        if not args:
            raise Exception("Missing arguments for class creation!")

        if args.max_tokens:
            self.max_tokens = args.max_tokens

        if args.max_test_instances:
            self.max_instances = args.max_test_instances

        self.model_name = model_name

        try:
            self.model_label = model_name.split("/")[1].replace("-", "_").replace(".", ",").lower()
        except Exception as error:
            logger.info(error)
            raise Exception("Invalid model name! (e.g. ibm/granite-13b-chat-v2)")

        self.watsonx_model = args.watsonx_model
        self.bam_model = args.bam_model
        self.overwrite_results = args.overwrite

        create_path(self.model_label)

    def get_model_parameters(self) -> tuple[str, int, bool, bool]:
        return self.model_name, self.max_tokens, self.watsonx_model, self.bam_model

    @abstractmethod
    def get_evaluation_name(self) -> str:
        """
        Return current evaluation name.
        """
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
        Implement the current model's evaluation.
        """
        pass
    
    def get_pretty_name(self) -> str:
        return f"{type(self).__name__}-{self.model_label}".lower()

    def get_results_path(self) -> str:
        return f'{self.model_label}/{self.results_folder}'

    def get_predictions_path(self) -> str:
        return f'{self.model_label}/{self.predictions_folder}'

    def result_file_exists(self, filename: str) -> bool:
        return file_exists(f'{self.get_results_path()}/{filename}')
    
    def prediction_file_exists(self, filename: str) -> bool:
        return file_exists(f'{self.get_predictions_path()}/{filename}')

    def save_predictions(self, filename: str, data: Dict[str, Any] | List[Dict[str, Any]] | Any) -> None:
        """
        Save model's prediticions into a pickle file. 

        No need to worry about file path, just the file name.
        """
        save_pickle(f'{self.get_predictions_path()}/{filename}', data)
    
    def load_predictions(self, filename: str) -> Any:
        pickle_file = open(f'{os.getenv("EVAL_HOME")}/{self.get_predictions_path()}/{filename}', 'rb')
        data = pickle.load(pickle_file)
        pickle_file.close()
        return data

    def save_results(self, filename: str, data: Dict[str, Any] | List[Dict[str, Any]] | Any) -> None:
        """
        Save model's inferences into a JSON file. 

        No need to worry about file path, just the file name.
        """
        save_json_file(f'{self.get_results_path()}/{filename}', data)

    def time_start(self) -> float:
        """
        Call this method to add a start time to your logs.

        Returns start time.
        """
        start_time = time.perf_counter()

        logger.info(f"[START] {type(self).__name__}: {start_time}")

        return start_time

    def time_end(self, start_time: float | None = None) -> None:
        """
        Call this method to add an end time to your logs.

        If a start time is provided, overall execution time will be logged.
        """
        if not start_time:
            logger.info(f"[END] {type(self).__name__}: {time.perf_counter()}")
        else:
            execution_time = str(datetime.timedelta(seconds=time.perf_counter()-start_time))

            logger.info("[EXECUTION] time: {}".format(execution_time))
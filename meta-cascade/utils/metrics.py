import json
import os
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from typing import Any

class ExtendedBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_config["protected_namespaces"] = ()

class GlobalScore(ExtendedBaseModel):
    accuracy: float | None = None
    accuracy_ci_high: float | None = None
    accuracy_ci_low: float | None = None
    llama_3_70b_instruct_parser: float = \
        Field(alias='llama_3_70b_instruct_ibm_genai_template_mixeval_multi_choice_parser')
    llama_3_70b_instruct_parser_ci_high: float = \
        Field(alias='llama_3_70b_instruct_ibm_genai_template_mixeval_multi_choice_parser_ci_high')
    llama_3_70b_instruct_parser_ci_low: float = \
        Field(alias='llama_3_70b_instruct_ibm_genai_template_mixeval_multi_choice_parser_ci_low')
    score: float
    score_ci_high: float
    score_ci_low: float
    score_name: str


class InstanceScore(ExtendedBaseModel):
    accuracy: float | None = None
    judge_raw_input: str
    judge_raw_output: str
    llama_3_70b_instruct_parser: float = \
        Field(alias='llama_3_70b_instruct_ibm_genai_template_mixeval_multi_choice_parser')
    score: float
    score_name: str


class Score(ExtendedBaseModel):
    global_: GlobalScore = Field(alias='global')
    instance: InstanceScore


class OutCast(ExtendedBaseModel):
    accuracy: float | None = None
    llmaj: float
    reference: str
    model_answer: str
    jugde_prompt: str
    judge_answer: str


class Datasets(ExtendedBaseModel):
    dataset_name: str
    instances_number: int
    discrepancies_number: int
    outcasts: list[OutCast]


class Results(ExtendedBaseModel):
    model_name: str
    results: list[Datasets]

class Subtask(ExtendedBaseModel):
    name: str
    accuracy: float
    llmaj: float

class Accuracies(ExtendedBaseModel):
    subtasks: list[Subtask]

def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

class MetricsChecker:
    def __init__(self) -> None:
        pass
    
    def __get_subtasks_data(self, results_folder_path: str):
        # Path structure should be in this format:
        # /path/to/model-name/results
        for path, _, files in os.walk(results_folder_path):
            for name in sorted(files):
                json_file = open(os.path.join(path, name))
                data = json.load(json_file)
                json_file.close()
                yield name, data


    def __get_model_accuracies(self, data: Any) -> tuple[float, float]:
        # Check if data has a global score for accuracy
        # i.e. 'metrics.accuracy' was added to the array
        # of metrics in UNITXT evaluation

        value: Score = Score(
            **data[0]['score']
        )

        accuracy_score_value: float = 0.0

        # Just in case
        if value.global_.accuracy is None and value.global_.llama_3_70b_instruct_parser is None:
            raise Exception('Unexpected scenario! No metrics accuracy found! (e.g. accuracy, llama_3_70b_instruct_parser)')

        if value.global_.accuracy is not None:
            accuracy_score_value = value.global_.accuracy
        else:
            calculated_accuracy: list[float] = []

            # Calculates subtask accuracy
            for instance in data:
                instance_accuracy: float = 0.0

                model_answer = instance['processed_prediction'][0].lower()
                reference_answer = instance['processed_references'][0].lower()

                # Checked if first letter of model's answer matches the reference answer
                if model_answer in ['a', 'b', 'c', 'd'] and model_answer == reference_answer:
                    instance_accuracy = 1.0

                calculated_accuracy.append(instance_accuracy)

            from sklearn.metrics import accuracy_score

            expected = [1.0] * len(calculated_accuracy)
            accuracy_score_value = accuracy_score(expected, calculated_accuracy)

        return accuracy_score_value, value.global_.llama_3_70b_instruct_parser


    def __get_instance_accuracies(self, instance: Any) -> tuple[float, float]:
        # Check if instance has a score for accuracy
        # i.e. 'metrics.accuracy' was added to the array
        # of metrics in UNITXT evaluation
        value: Score = Score(
            **instance['score']
        )

        # Just in case
        if value.instance.accuracy is None and value.instance.llama_3_70b_instruct_parser is None:
            raise Exception('Unexpected scenario! No metrics accuracy found! (e.g. accuracy, llama_3_70b_instruct_parser)')

        instance_accuracy: float = 0.0

        if value.instance.accuracy is not None:
            instance_accuracy = value.instance.accuracy
        else:
            model_answer = instance['processed_prediction'][0].lower()
            reference_answer = instance['processed_references'][0].lower()

            # Checked if first letter of model's answer matches the reference answer
            if model_answer in ['a', 'b', 'c', 'd'] and model_answer == reference_answer:
                instance_accuracy = 1.0

        return instance_accuracy, value.instance.llama_3_70b_instruct_parser


    def __calculate_outcasts(self, data: Any) -> list[OutCast] | None:
        outcasts: list[OutCast] = []

        discrepancy: bool = False

        if len(data) > 0:
            accuracy, llmaj = self.__get_model_accuracies(data)

            # Check for discrepancy between global scores
            discrepancy = accuracy != llmaj
        else:
            return None

        if discrepancy:
            for instance in data:
                accuracy, llmaj = self.__get_instance_accuracies(instance)

                value: Score = Score(
                    **instance['score']
                )

                if accuracy != llmaj:
                    outcasts.append(
                        OutCast(
                            accuracy=accuracy,
                            llmaj=llmaj,
                            reference=instance['processed_references'][0],
                            jugde_prompt=value.instance.judge_raw_input,
                            model_answer=instance['processed_prediction'],
                            judge_answer=value.instance.judge_raw_output
                        )
                    )

        return outcasts if len(outcasts) > 0 else None


    def check_model_accuracies(self, results_folder_path: str) -> Accuracies | None:
        result: Accuracies = Accuracies(
            subtasks=[]
        )

        for filename, data in self.__get_subtasks_data(results_folder_path):
            subtask_name = filename.split('.')[0]

            if len(data) > 0:
                accuracy, llmaj = self.__get_model_accuracies(data)
                
                if accuracy != llmaj:
                    result.subtasks.append(
                        Subtask(
                            name=subtask_name,
                            accuracy=accuracy,
                            llmaj=llmaj
                        )
                    )

        return result

    def check_model_results(self, results_folder_path: str, save_json: bool = False) -> str:
        overall_result: Results = Results(
            model_name='',
            results=[]
        )

        model_name = results_folder_path.split('/')[-2]
        overall_result.model_name = model_name

        for subtask_filename, data in self.__get_subtasks_data(results_folder_path):
            result = self.__calculate_outcasts(data)

            if result and len(result) > 0:
                overall_result.results.append(
                    Datasets(
                        dataset_name=subtask_filename,
                        instances_number=len(data),
                        discrepancies_number=len(result),
                        outcasts=result
                    )
                )
    
        if len(overall_result.results) > 0:
            if save_json:
                save_to = '/'.join(results_folder_path.split('/')[:-1])
                filename = f'{save_to}/{model_name}-results.json'

                if save_json:
                    for result in overall_result.results:
                        with open(f'{filename}-{result.dataset_name}', "w") as outfile:
                            serializable_data = json.dumps(result.model_dump(), default=handle_non_serializable)
                            outfile.write(serializable_data)
                            outfile.close()

                return f'Check results saved to: {filename}'
            else:
                return f'Discrepancies found for {len(overall_result.results)} subtasks!'
        else:
            return f'No discrepancies found for {model_name}!'

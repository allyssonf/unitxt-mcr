import json
import logging
import os

from pydantic import BaseModel, Field, ConfigDict
from .files import handle_non_serializable
from typing import Any

logger = logging.getLogger(__name__)

class GlobalScore(BaseModel):
    accuracy: float
    accuracy_ci_high: float
    accuracy_ci_low: float
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


class InstanceScore(BaseModel):
    accuracy: float
    judge_raw_input: str
    judge_raw_output: str
    llama_3_70b_instruct_parser: float = \
        Field(alias='llama_3_70b_instruct_ibm_genai_template_mixeval_multi_choice_parser')
    score: float
    score_name: str


class Score(BaseModel):
    global_: GlobalScore = Field(alias='global')
    instance: InstanceScore


class OutCast(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    instance_index: int
    accuracy: float
    llmaj: float
    reference: str
    model_answer: str
    jugde_prompt: str
    judge_answer: str


class Datasets(BaseModel):
    dataset_name: str
    discrepancies: int
    outcasts: list[OutCast]


class Results(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    results: list[Datasets]


class ResultsChecker:
    def __init__(self) -> None:
        pass

    def __calculate_outcasts(self, data: Any) -> list[OutCast] | None:
        outcasts: list[OutCast] = []

        discrepancy: bool = False

        if len(data) > 0:
            value: Score = Score(
                **data[0]['score']
            )

            discrepancy = value.global_.accuracy != value.global_.llama_3_70b_instruct_parser
        else:
            return None

        if discrepancy:
            for index, instance in enumerate(data):
                value: Score = Score(
                    **instance['score']
                )

                if value.instance.accuracy != value.instance.llama_3_70b_instruct_parser:
                    outcasts.append(
                        OutCast(
                            instance_index=index,
                            accuracy=value.instance.accuracy,
                            llmaj=value.instance.llama_3_70b_instruct_parser,
                            reference=instance['processed_references'][0],
                            jugde_prompt=value.instance.judge_raw_input,
                            model_answer=instance['processed_prediction'],
                            judge_answer=value.instance.judge_raw_output
                        )
                    )

        return outcasts if len(outcasts) > 0 else None

    def check_results(self, results_folder_path: str) -> str:
        logger.info('Running LLMaJ results check.')

        overall_result: Results = Results(
            model_name='',
            results=[]
        )

        model_result_path = ""
        model_name = ""

        for path, _, files in os.walk(results_folder_path):
            # Path structure should be in this format:
            # /path/to/model-name/results
            model_result_path = str(path)
            model_name = model_result_path.split('/')[-2]

            if len(overall_result.model_name) == 0:
                overall_result.model_name = model_name

            for name in files:
                json_file = open(os.path.join(path, name))
                data = json.load(json_file)
                result = self.__calculate_outcasts(data)

                if result:
                    overall_result.results.append(
                        Datasets(
                            dataset_name=name,
                            discrepancies=len(result),
                            outcasts=result
                        )
                    )
    
        if len(overall_result.results) > 0:
            logger.info(f'Model: {model_name}')
            logger.info(f'\tNumber of subtasks with discrepancy: {len(overall_result.results)}')

            wrong_answers: int = 0
            for dataset in overall_result.results:
                wrong_answers += len(dataset.outcasts)
            
            logger.info(f'\tNumber of wrong answers: {len(overall_result.results)}')

            save_to = '/'.join(model_result_path.split('/')[:-1])
            filename = f'{save_to}/{model_name}-results.json'

            with open(filename, "w") as outfile: 
                serializable_data = json.dumps(overall_result.model_dump(), default=handle_non_serializable)
                outfile.write(serializable_data)

            return f'Check results saved to: {filename}'
        else:
            return f'No discrepancies found for {model_name}!'

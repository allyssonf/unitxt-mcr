import json
import os
from pydantic import BaseModel, Field, ConfigDict
# from utils.files import handle_non_serializable
from typing import Any


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
    accuracy: float
    llmaj: float
    reference: str
    model_answer: str
    jugde_prompt: str
    judge_answer: str


class Datasets(BaseModel):
    dataset_name: str
    outcasts: list[OutCast]


class Results(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    results: list[Datasets]

def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

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
            for instance in data:
                value: Score = Score(
                    **instance['score']
                )

                if value.instance.accuracy != value.instance.llama_3_70b_instruct_parser:
                    outcasts.append(
                        OutCast(
                            accuracy=value.instance.accuracy,
                            llmaj=value.instance.llama_3_70b_instruct_parser,
                            reference=instance['processed_references'][0],
                            jugde_prompt=value.instance.judge_raw_input,
                            model_answer=instance['processed_prediction'],
                            judge_answer=value.instance.judge_raw_output
                        )
                    )

        return outcasts if len(outcasts) > 0 else None

    def check_results(self, results_folder_path: str, save_json: bool = False) -> str:
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
            overall_result.model_name = model_name

            for name in files:
                json_file = open(os.path.join(path, name))
                data = json.load(json_file)
                result = self.__calculate_outcasts(data)

                if result:
                    overall_result.results.append(
                        Datasets(
                            dataset_name=name,
                            outcasts=result

                        )
                    )
    
        if len(overall_result.results) > 0:
            print(f'Model: {model_name}')
            print(f'\tNumber of subtasks with discrepancy: {len(overall_result.results)}')

            wrong_answers: int = 0
            for dataset in overall_result.results:
                wrong_answers += len(dataset.outcasts)
            
            print(f'\tNumber of wrong answers: {len(overall_result.results)}')

            save_to = '/'.join(model_result_path.split('/')[:-1])
            filename = f'{save_to}/{model_name}-results.json'

            if save_json:
                with open(filename, "w") as outfile: 
                    serializable_data = json.dumps(overall_result.model_dump(), default=handle_non_serializable)
                    outfile.write(serializable_data)

            return f'Check results saved to: {filename}'
        else:
            return f'No discrepancies found for {model_name}!'


results_checker: ResultsChecker = ResultsChecker()

models_list = [
    'granite_13b_chat_v2',
    'granite_34b_code_instruct',
    'llama_3_405b_instruct',
    'llama_3_70b_instruct',
    'llama_3_8b_instruct',
    'mistral_large',
    'mixtral_8x7b_instruct_v01'
]

for model in models_list:
    results_path = f'/data/home/allysson/Data/1_mmlu_new/{model}/results'
    results_checker.check_results(results_path)
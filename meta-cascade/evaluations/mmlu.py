import logging
import gc
import torch
import os

from interfaces.evaluation import Evaluation
from interfaces.models import ModelLoader

from unitxt.api import evaluate, load_dataset

logger = logging.getLogger(__name__)

class Mmlu(Evaluation):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)

    def get_evaluation_name(self) -> str:
        return 'MMLU'

    def evaluate(self):
        subtasks = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]

        model_loader = ModelLoader()

        for sub in subtasks:
            start_time = self.time_start()
            prediction_file = f'mmlu_{sub}.pkl'
            result_file = f'mmlu_{sub}.json'

            if not self.result_file_exists(result_file) or self.overwrite_results:
                evaluation = f"card=cards.mmlu.{sub}"
                evaluation += f", template_card_index=0, metrics=[{os.getenv('EVAL_METRICS')}]"

                if self.max_instances:
                    evaluation += f", max_test_instances={self.max_instances}"

                logger.info(">>>>>> EVALUATING <<<<<<")
                logger.info("\n")
                logger.info(evaluation)
                logger.info("\n")

                dataset = load_dataset(evaluation)

                test_dataset = dataset["test"]

                name, tokens, watsonx = self.get_model_parameters()

                inference_model = model_loader.get_inference_model(model_name=name, watsonx=watsonx, max_tokens=tokens)

                predictions = []

                if not self.prediction_file_exists(prediction_file):
                    predictions = inference_model.infer(test_dataset)
                else:
                    logger.info(f'Loading predictions from file: {prediction_file}')
                    predictions = self.load_predictions(prediction_file)

                self.save_predictions(prediction_file, predictions)

                evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

                self.save_results(result_file, evaluated_dataset)

                # Clear cache
                gc.collect()
                torch.cuda.empty_cache()
            else:
                logger.info(f'Subtask {sub} already processed!')

            self.time_end(start_time)

        return


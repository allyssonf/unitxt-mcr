import logging
from interfaces.evaluation import Evaluation

from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFPipelineBasedInferenceEngine

logger = logging.getLogger(__name__)

class Mmlu(Evaluation):
    model_name: str | None = None
    max_tokens: int = 32
    max_instances: int | None = None

    def __init__(self, model_name: str, max_tokens: int | None, max_instances: int | None):
        if max_tokens:
            self.max_tokens = max_tokens

        if max_instances:
            self.max_instances = max_instances

        self.model_name = model_name

    def evaluate(self):
        # For now, just abstract algebra
        subtasks = [
            "abstract_algebra"
        ]

        for idx, sub in enumerate(subtasks):
            evaluation = f"card=cards.mmlu.{sub}, template_card_index=0, metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_mixeval_multi_choice_parser]"

            if self.max_instances:
                evaluation += f", max_test_instances={self.max_instances}"

            logger.info(">>>>>> EVALUATING <<<<<<")
            logger.info("\n")
            logger.info(evaluation)
            logger.info("\n")

            dataset = load_dataset(evaluation)

            test_dataset = dataset["test"]

            inference_model = HFPipelineBasedInferenceEngine(
                model_name=self.model_name, max_new_tokens=self.max_tokens
            )

            predictions = inference_model.infer(test_dataset)

            model_label=self.model_name.split("/")[1].replace("-", "_").replace(".", ",").lower()

            self.save_predictions(f'predictions/{model_label}_mmlu_{sub}.pkl', predictions)

            evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

            self.save_results(f'results/{model_label}_mmlu_{sub}.json', evaluated_dataset)

        return


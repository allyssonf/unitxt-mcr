import time
import logging
from interfaces.evaluation import Evaluation

from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFPipelineBasedInferenceEngine

logger = logging.getLogger(__name__)

class MMLU_Llama3_8b_Instruct(Evaluation):
    def __init__(self):
        pass

    def evaluate(self):
        # For now, just abstract algebra
        subtasks = [
            "abstract_algebra"
        ]

        model_name="meta-llama/Meta-Llama-3-8B-Instruct"

        for idx, sub in enumerate(subtasks):
            evaluation = f"card=cards.mmlu.{sub}, template_card_index=0, metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_mixeval_multi_choice_parser], max_test_instances=5"

            logger.info(">>>>>> EVALUATING <<<<<<")
            logger.info("\n")
            logger.info(evaluation)
            logger.info("\n")

            dataset = load_dataset(evaluation)

            test_dataset = dataset["test"]

            # model_inputs = test_dataset["source"]

            inference_model = HFPipelineBasedInferenceEngine(
                model_name=model_name, max_new_tokens=32
            )

            predictions = inference_model.infer(test_dataset)

            self.save_predictions('predictions/pred_mmlu_abstract_algebra.pkl', predictions)

            evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

            self.save_results(f'results/{int(time.time())}_mmlu_{sub}_result.json', evaluated_dataset)

        return


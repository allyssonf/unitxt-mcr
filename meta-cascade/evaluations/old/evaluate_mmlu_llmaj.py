import time
import sys
import evaluations.utils.files as utilities

from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFPipelineBasedInferenceEngine


# For now, just abstract algebra
subtasks = [
    "abstract_algebra"
]

model_name="meta-llama/Meta-Llama-3-8B-Instruct"

for idx, sub in enumerate(subtasks):
    evaluation = f"card=cards.mmlu.{sub}, template_card_index=0, metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn]"

    print(">>>>>> EVALUATING <<<<<<")
    print("\n")
    print(evaluation)
    print("\n")

    dataset = load_dataset(evaluation)

    test_dataset = dataset["test"]

    print(test_dataset)

    # import sys

    # sys.exit(1)

    model_inputs = test_dataset["source"]

    print(">>>>>> INPUTS <<<<<<")
    print("\n")
    print(model_inputs)
    print("\n")

    inference_model = HFPipelineBasedInferenceEngine(
        model_name=model_name, max_new_tokens=32
    )

    predictions = inference_model.infer(test_dataset)

    utilities.save_pickle('predictions/pred_mmlu_abstract_algebra.pkl', predictions)

    import sys

    sys.exit(1)

    print(">>>>>> PREDICTIONS <<<<<<")
    print("\n")
    print(predictions)
    print("\n")

    # sys.exit(0)

    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

    file_path = f"{int(time.time())}_mmlu_{sub}_result.json"

    file_name = f"/data/home/allysson/Projects/LLMEval/unitxt-mcr/evaluations/data/{file_path}"

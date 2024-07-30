import json
import numpy as np
import time
import sys

from unitxt import get_from_catalog
from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.templates import TemplatesList

def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

# Retrieve templates list
templates_list: TemplatesList = get_from_catalog("templates.qa.multiple_choice.with_topic.paraphrase.mmlu.all")

# Arbitrary choice
template = templates_list.items[0].get_pretty_print_name()

# For now, just abstract algebra
subtasks = [
    "abstract_algebra"
]

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# model_name="meta-llama/Meta-Llama-3-70B-Instruct"

for idx, sub in enumerate(subtasks):
    evaluation = f"card=cards.mmlu.{sub}, template_card_index=0, metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_mixeval_multi_choice_parser], max_test_instances=5"

    print(">>>>>> EVALUATING <<<<<<")
    print("\n")
    print(evaluation)
    print("\n")

    dataset = load_dataset(evaluation)

    test_dataset = dataset["test"]

    model_inputs = test_dataset["source"]

    print(">>>>>> INPUTS <<<<<<")
    print("\n")
    print(model_inputs)
    print("\n")

    inference_model = HFPipelineBasedInferenceEngine(
        model_name=model_name, max_new_tokens=32
    )

    predictions = inference_model.infer(test_dataset)

    print(">>>>>> PREDICTIONS <<<<<<")
    print("\n")
    print(predictions)
    print("\n")

    # sys.exit(0)

    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

    file_path = f"{int(time.time())}_mmlu_{sub}_{template}_result.json"

    file_name = f"/data/home/allysson/Projects/LLMEval/unitxt-mcr/mcr-evaluations/data/{file_path}"
    
    with open(file_name, "w") as outfile: 
        serializable_data = json.dumps(evaluated_dataset, default=handle_non_serializable)
        outfile.write(serializable_data)

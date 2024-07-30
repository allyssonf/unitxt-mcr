from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge

model = "meta-llama/llama-3-70b-instruct"
format = "formats.llama3_instruct"
template = "templates.response_assessment.rating.mixeval_multi_choice_parser"

inference_model = IbmGenAiInferenceEngine(model_name=model, max_new_tokens=252)
model_label = model.split("/")[1].replace("-", "_").replace(".", ",").lower()
model_label = f"{model_label}_ibm_genai"
template_label = template.split(".")[-1]
metric_label = f"{model_label}_template_{template_label}"
metric = LLMAsJudge(
    inference_model=inference_model,
    template=template,
    task="rating.single_turn_with_reference",
    format=format,
    main_score=metric_label,
    prediction_type="str",
)

add_to_catalog(
    metric,
    f"metrics.llm_as_judge.rating.{model_label}_template_{template_label}",
    overwrite=True,
)


from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate

# Use the same processor of MT Bench to extract the result
        # "Options:\n{choices}\n"
add_to_catalog(
    InputOutputTemplate(
        instruction="In this task, I want you to act as an option extractor.\nYou will be provided"
            " with a multiple-choice question, its options, and the model’s answer, while"
            " the context of the question is not given here. Your task is to extract or judge"
            " which option is chosen by the model based on its response, without seeing the"
            " context of the question. The extracted option should be one of the provided option"
            " letters. Your should first briefly give your reasoning process, and then give" 
            " the extracted option letter. The extracted option must strictly follow this "
            ' format: "[[option letter]]", e.g., "The option chosen by the model: [[A]]". \n\n',
        input_format="Question: {question}\n"
        "Model's answer\n{answer}\n"
        "Your judgement:",
        output_format="[[{rating}]]",
        postprocessors=[
            r"processors.extract_mt_bench_rating_judgment",
        ],
    ),
    "templates.response_assessment.rating.mixeval_multi_choice_parser",
    overwrite=True,
)

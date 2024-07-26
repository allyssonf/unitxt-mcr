from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate

# Use the same processor of MT Bench to extract the result
        # "Options:\n{choices}\n"
add_to_catalog(
    InputOutputTemplate(
        instruction="In this task, I want you to act as a judge checking if a opinion is right or wrong."
            " \nYou will be provided with a multiple-choice question, the golden answer, and the model’s answer,"
            " while the context of the question is not given here. Your task is to extract and judge"
            " the option chosen by the model based on its response, without seeing the context of the question." 
            " The option should be one of the provided option letters in the question. You should first briefly give"
            ' your reasoning process, and then give a rate "[[10]]" if the answer is right, and "[[0]]" if the answer is wrong.'
            ' Your judgement score should strictly follow this format: "[[rate]]", e.g. "My verdict is: [[rate]]". \n\n',
        input_format="Question: {question}\n"
        "Golden Answer: {reference_answer}\n"
        "Model's Answer: {answer}\n"
        "Your judgement:",
        output_format="[[{rating}]]",
        postprocessors=[
            r"processors.extract_mt_bench_rating_judgment",
        ],
    ),
    "templates.response_assessment.rating.mixeval_multi_choice_parser",
    overwrite=True,
)

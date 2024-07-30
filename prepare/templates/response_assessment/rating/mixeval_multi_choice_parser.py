from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate

# Use the same processor of MT Bench to extract the result
add_to_catalog(
    InputOutputTemplate(
        instruction="A task was given to a model, and the task as well as the model's response is as below."
            " There's also the Golden response. Compare the model's response and golden response."
            " Act as a judge and give a rating on [[10]] if the model's response matches the golden"
            " response otherwise give a rating of [[0]].\n\n"
            "Strictly follow the output format below and maintain the square brackets.\n"
            '```\n"rating": "[[your-rating]]"\n```\n\n'
            ,
        input_format="Task: {question}\n"
        "Model's Response: {answer}\n"
        "Golden Answer: {reference_answer}\n\n",
        output_format="[[{rating}]]",
        postprocessors=[
            r"processors.extract_mt_bench_rating_judgment",
        ],
    ),
    "templates.response_assessment.rating.mixeval_multi_choice_parser",
    overwrite=True,
)

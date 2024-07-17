from unitxt.catalog import add_to_catalog
from unitxt.templates import MultipleChoiceTemplate, TemplatesList

paraphrase_templates = {
    "with_topic": {
        "mmlu": [
            "The following are multiple choice questions (with answers) about {topic}.\n{question}\nAnswers:\n{choices}\nAnswer:",
            "Here are some questions (with solutions) regarding {topic}.\n{question}\nSolutions:\n{choices}\nSolution:",
            "Below are several questions (along with responses) about {topic}.\n{question}\nResponses:\n{choices}\nResponse:",
            "Here are a few questions (and their corresponding answers) on {topic}.\n{question}\nAnswers:\n{choices}\nAnswer:",
            "The following are a series of questions (and solutions) about {topic}.\n{question}\nSolutions:\n{choices}\nSolution:",
            "Here are some questions (with their respective answers) on {topic}.\n{question}\nAnswers:\n{choices}\nAnswer:",
            "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}\nAnswers:\n{choices}\nAnswer:",
            "Here are some multiple-choice questions (along with their answers) regarding {topic}.\nQuestion: {question}\nOptions:\n{choices}\nCorrect answer:",
            "Below are several multiple-choice queries (and solutions) about {topic}.\nQuestion: {question}\nAlternatives:\n{choices}\nRight response:",
            "Here are a series of multiple-choice inquiries (accompanied by answers) on {topic}.\nQuestion: {question}\nResponses:\n{choices}\nCorrect answer:",
            "Here are a few multiple-choice questions (together with the correct answers) on {topic}.\nQuestion: {question}\nOptions:\n{choices}\nCorrect response:",
            "Here are a set of multiple-choice questions (and their answers) about {topic}.\nQuestion: {question}\nChoices:\n{choices}\nAnswer:",
            "The following are multiple choice questions (with answers) about {topic}.\n\n{question}\n{choices}\nAnswer:",
            "Here are some questions (along with answers) regarding {topic}.\n\n{question}\n{choices}\nCorrect response:",
            "Here are several questions (and their respective answers) about {topic}.\n\n{question}\n{choices}\nAnswer:",
            "Here are some multiple-choice questions (and their solutions) on {topic}.\n\n{question}\n{choices}\nAnswer:",
            "Here are a few questions (together with their answers) about {topic}.\n\n{question}\n{choices}\nAnswer:",
            "Here are some questions (accompanied by their answers) concerning {topic}.\n\n{question}\n{choices}\nAnswer:"
        ] 
    }
}

template_handles = []

for template_type, template_group in paraphrase_templates.items():
    for benchmark_name, inputs in template_group.items():
        for index, input_format in enumerate(inputs):
            template = MultipleChoiceTemplate(
                input_format=input_format,
                target_field="answer",
                choices_separator="\n",
                target_choice_format=" {choice_numeral}" if "lm_eval_harness"
                in benchmark_name else "{choice_numeral}",
                postprocessors=["processors.first_character"],
            )

            template_handle = f"templates.qa.multiple_choice.{template_type}.paraphrase.variation_{index}"
            template_handles.append(template_handle)

            add_to_catalog(
                template,
                template_handle,
                overwrite=True,
            )

add_to_catalog(
    artifact=TemplatesList(
        template_handles
    ),
    name="templates.qa.multiple_choice.with_topic.paraphrase.mmlu.all",
    overwrite=True,
)

# Follows the guides at:
#   https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
#   https://llama.meta.com/docs/how-to-guides/prompting/

def generate_prompt(question, model_answer, golden_response):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

A task was given to a model with a list of options as possible answers, there's also the golden response. Act as a judge comparing the model's response and the golden response and give a rating of [[10]] if the model's response matches the golden response otherwise provide a rating of [[0]]. Below are some examples. 

Strictly follow the output format below and maintain the double square brackets.

```
rating: [[your-rating]]
```

The task, the model's response chosen from the options list, and the golden response are as follows:

Task: {question}

Model's Response: {model_answer}

Golden Response: {golden_response}

Your Response: <|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    return prompt


def generate_new_prompt(question, model_answer, golden_answer):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

I want you to act as a judge for how well a model did answering a user-defined task.

You will be provided with a user-defined task that was given to the model, its golden answer(s), and the model's answer. The context of the task may not be given here. Your task is to judge how correct is the model's answer. Your task is to judge how correct the model’s answer is based on the golden answer(s), without seeing the context of the task, and then give a correctness score. The correctness score should be one of the below numbers: 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). You should also add a brief justification regarding how the model’s answer conforms to or contradicts the golden answer(s). Your response must follow the format
{{
   "correctness_score": your_score,
   "justification": your_justification
}}


Below are some examples:

Example 1:

User-defined task --- Sandy bought 1 million Safe Moon tokens. She has 4 siblings. She wants to keep half of them
to herself and divide the remaining tokens among her siblings. After splitting it up, how many more tokens
will she have than any of her siblings?
Golden Answer(s) --- <answer 1> 375000
Model’s Answer --- Sandy will have more tokens than any sibling by 3/8 million.
Your Response ---
{{
    "correctness_score": 1.0,
    "justification": "The golden answer states that Sandy will have 375,000 more tokens than any of her siblings, which is a precise numerical value. The model’s answer translates this scenario into a fraction of the total, saying Sandy will have more tokens than any sibling by 3/8 million. 1 million tokens * 3/8 = 375,000 tokens. So the model provided an answer in fractional form that, when converted to a numerical value, exactly matches the golden answer’s quantity."
}}

Example 2:

User-defined task --- Extract the names and emails mentioned in the following text.

Richard Matthew Stallman (/ˈstɔːlmən/ STAWL-mən; born March 16, 1953), also known by his initials, rms,[1] is an American free software movement activist and programmer. He can be reached at rms@gnu.org based on the information on his webpage.
Golden Answer(s) --- {{"names":  ["Richard Matthew Stallman"], "emails": ["rms@gnu.org"]}}
Model’s Answer --- names: Richard Matthew Stallman
Your Response:
{{
    "correctness_score": 0.5,
    "justification": "The model correctly identified the names but failed to extract the emails."
}}

Example 3:

User defined task --- Question: In 2015 Edgar Lungu became prime minister of?
Golden Answer: <answer 1> Zambia; <answer 2> Zamibia; <answer 3> People of Zambia; <answer
4> Zambian cuisine; <answer 5> Zambians; <answer 6> Culture of Zambia; <answer 7> Etymology of
Zambia; <answer 8> Zambia; <answer 9> Health care in Zambia; <answer 10> ISO 3166-1:ZM; <answer
11> Republic Of Zambia; <answer 12> Cuisine of Zambia; <answer 13> Sport in Zambia; <answer 14>
Republic of Zambia; <answer 15> Zambian people; <answer 16> Name of Zambia
Model’s Answer: Prime Minister
Your Response:
{{
    "correctness_score": 0.0,
    "justification": "The golden answers provide a detailed list of entities all relating to Zambia, indicating that Edgar Lungu became the leader (specifically, they mentioned \"prime minister\") of Zambia in 2015. The model’s answer, \"Prime Minister,\" merely repeats part of the question without answering it."
}}

Note that each one of the golden answers is considered correct. Thus if the Model’s Answer matches any
one of the golden answers, it should be considered correct. 

Now judge the below case, and provide your response.
User defined task --- {question}
Golden Answer(s) --- {golden_answer}
Model’s Answer --- {model_answer}
Your Response: <|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    return prompt

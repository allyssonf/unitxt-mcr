import gradio as gr

import os
import sys

# Allow loading function from project
module_path = os.path.abspath(os.path.join('../unitxt-mcr/meta-cascade'))
print(module_path)
sys.path.insert(0, module_path)

from utils.metrics import MetricsChecker, Accuracies

models_list = [
    'ibm/granite-13b-chat-v2',
    'ibm/granite-34b-code-instruct',
    'meta-llama/llama-3-70b-instruct',
    'meta-llama/llama-3-405b-instruct',
    'meta-llama/llama-3-8b-instruct',
    'mistralai/mistral-large',
    'mistralai/mixtral-8x7b-instruct-v01',
    'Qwen/Qwen2-72B-Instruct',
    'CohereForAI/c4ai-command-r-plus',
    'microsoft/Phi-3-medium-4k-instruct',
    'microsoft/Phi-3-mini-4k-instruct',
    '01-ai/Yi-1.5-34B-Chat',
]


def plot_accuracy_data(result: Accuracies, model_name: str):
    import matplotlib.pyplot as plt
    import numpy as np

    print('Plot accuracy data...')

    accuracy = [subtask.accuracy for subtask in result.subtasks]

    llmaj = [subtask.llmaj for subtask in result.subtasks]

    names = [subtask.name.split('mmlu_')[1].replace("_", " ") for subtask in result.subtasks]

    plt.subplots(figsize =(16, 7))

    bar_width = 0.4

    x1 = np.arange(len(accuracy))
    x2 = [x + bar_width for x in x1]
    plt.bar(x1, accuracy, align='edge', width=bar_width, color='#fdb06d', label='Accuracy Metric')
    plt.bar(x2, llmaj, align='edge', width=bar_width, color='#6dbafd', label='LLMaJ Metric')
    plt.xticks([r + bar_width for r in range(len(names))], names, rotation=90)


    plt.legend()

    plt.title(f"\nUNITXT Accuracy x LLMaJ - {model_name}\n\n{'NO DATA FOUND!\n\n' if len(accuracy) == 0 else ''}")
    plt.xlabel("MMLU Subtasks")
    plt.ylabel("Subtask Accuracy")

    plt.grid(axis='y', linewidth=0.2)

    return plt

def generate_plot(model_name: str | None, result_path: str | None):
    print('Generating plot...')

    if model_name is None or result_path is None:
        print('Missing required fields')
        return

    results_checker = MetricsChecker()

    model_only_name = model_name.split('/')[1].replace('-', '_')

    model_full_path = f'{result_path}/{model_only_name}/results'

    results = results_checker.check_model_accuracies(model_full_path)

    return plot_accuracy_data(results, model_only_name)

    
with gr.Blocks() as server:
    gr.Markdown("## Meta Cascade Routing Model's Results")

    result_path = gr.Textbox(label="Path for Model's evaluation results folder:", placeholder="/path/to/evaluation/results/base/folder (.e.g. same value of EVAL_HOME env variable.)")

    model_selection = gr.Dropdown(models_list, label="Please select a model to check the results:")

    generate_button = gr.Button("Generate Visualization")

    gr.Markdown(f"### Model Results")

    figure = gr.Plot()

    generate_button.click(generate_plot, [model_selection, result_path], figure)


server.launch()
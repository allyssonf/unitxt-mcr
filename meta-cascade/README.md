# Meta Cascade UNITXT

## Installation

In unitxt-mcr root folder, run the following commands

> pip install -e .

> pip install ibm-generative-ai

> pip install setuptools

> pip install torch

> pip install transformers

> pip install accelerate

> pip install flash-attention

> pip install python-dotenv

> pip install ibm-watsonx-ai

> pip install pyairtable

### For some microsoft models:
> pip install flash-attn --no-build-isolation

## Running evaluation

Please set `EVAL_HOME` environment variable to the folder where results will be saved

```text
📦granite_13b_chat_v2
 ┣ 📂predictions
 ┃ ┗ 📜pred_mmlu_abstract_algebra.pkl
 ┣ 📂results
 ┃ ┣ 📜1721234793_mmlu_abstract_algebra_result.json
 ┃ ┣ 📜1721235069_mmlu_abstract_algebra_result.json
 ┃ ┗ 📜1721235212_mmlu_abstract_algebra_result.json
```


## Others

### Cleaning a python virtualenv (pyenv)

> pip uninstall unitxt

> pip freeze | xargs pip uninstall -y

### Running llama-3-70B model

#### 2 GPUs
HF_TOKEN=<TOKEN> CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B --host 0.0.0.0 --port 8095 --trust-remote-code --enforce-eager --tensor-parallel-size 2 --gpu-memory-utilization 1

#### 4 GPUs
HF_TOKEN=<TOKEN> CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --host 0.0.0.0 --port 8095 --trust-remote-code --tensor-parallel-size 4
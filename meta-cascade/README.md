# Meta Cascade UNITXT

## Installation

In unitxt root folder, run the following commands

> pip install -e .

> pip install setuptools

> pip install torch

> pip install transformers

> pip install accelerate

## Running evaluation

Please set `EVAL_HOME` environment variable to the folder where results will be saved

```text
📦data
 ┣ 📂predictions
 ┃ ┗ 📜pred_mmlu_abstract_algebra.pkl
 ┣ 📂results
 ┃ ┣ 📜1721234793_mmlu_abstract_algebra_result.json
 ┃ ┣ 📜1721235069_mmlu_abstract_algebra_result.json
 ┃ ┗ 📜1721235212_mmlu_abstract_algebra_result.json
```

Then run the following command (unitxt root folder):

>  EVAL_HOME=<DATA_FOLDER> GENAI_KEY=<YOUR_KEY> HF_TOKEN=<HF_TOKEN> HF_HOME=<HF_HOME> python meta-cascade/main.py  

## Others

### Cleaning a python virtualenv (pyenv)

> pip uninstall unitxt

> pip freeze | xargs pip uninstall -y

### Running llama-3-70B model

#### 2 GPUs
HF_TOKEN=<TOKEN> CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B --host 0.0.0.0 --port 8095 --trust-remote-code --enforce-eager --tensor-parallel-size 2 --gpu-memory-utilization 1

#### 4 GPUs
HF_TOKEN=<TOKEN> CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --host 0.0.0.0 --port 8095 --trust-remote-code --tensor-parallel-size 4
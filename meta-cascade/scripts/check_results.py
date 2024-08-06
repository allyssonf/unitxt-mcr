# models_list = [
#     'granite_13b_chat_v2',
#     'granite_34b_code_instruct',
#     'llama_3_405b_instruct',
#     'llama_3_70b_instruct',
#     'llama_3_8b_instruct',
#     'mistral_large',
#     'mixtral_8x7b_instruct_v01'
# ]

models_list = [
    'granite_34b_code_instruct'
]

import os
import sys

module_path = os.path.abspath(os.path.join('../unitxt-mcr/meta-cascade'))
print(module_path)
sys.path.insert(0, module_path)

import utils.metrics

for model in models_list:
    results_path = f'/data/home/allysson/EVAL_DATA/Tests/postprocessor_new/{model}/results'

    mc_handler = utils.metrics.MetricsChecker()

    print(mc_handler.check_model_results(results_folder_path=results_path, save_json=True))
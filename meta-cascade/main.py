import logging
import argparse
import time
import os
import gc

from evaluations.mmlu import Mmlu
from interfaces.evaluation import Evaluation
from utils.files import create_path
from dotenv import load_dotenv
from utils.airtable import AirTableLogger, Result
from utils.result_checker import ResultsChecker

load_dotenv(dotenv_path=f'{os.getcwd()}/meta-cascade')

logger = logging.getLogger(__name__)

def main(args):
    if not args.watsonx_model:
        if not os.getenv("CUDA_VISIBLE_DEVICES"):
            raise Exception("CUDA_VISIBLE_DEVICES env variable is needed to run local inference!")

    models_list = args.models_list.split(',')

    for model_name in models_list:
        new_evaluation = Mmlu(model_name, args)

        if not issubclass(type(new_evaluation), Evaluation):
            raise Exception("Subclass of Evaluation interface expected!")

        airtable_logger = AirTableLogger()

        airtable_logger.log_start(model_name, new_evaluation.get_evaluation_name())

        pretty_name = new_evaluation.get_pretty_name()

        create_path(f'{os.getcwd()}/logs', ignore_home=True)

        log_filename = f"./logs/{int(time.time())}_{pretty_name}_execution.log"

        logging.basicConfig(
            filename=log_filename, level=logging.INFO
        )

        evaluation_result: Result = Result.success
        result_path = f'{os.getenv("EVAL_HOME")}/{new_evaluation.get_results_path()}'
        evaluation_message: str = f'Results saved to: {result_path}\n'

        try:
            # Exceptions might arise while processing model's evaluation
            new_evaluation.evaluate()
            results_checker: ResultsChecker = ResultsChecker()
            evaluation_message += results_checker.check_results(result_path)

        except Exception as error:
            logger.info(error)
            logger.info(f'Error evaluating model {new_evaluation.get_pretty_name()}')
            evaluation_result = Result.failure
            evaluation_message = str(error)
            evaluation_message += f'\nMore info: {log_filename}'

        airtable_logger.log_end(evaluation_result, evaluation_message)

        del new_evaluation
        del airtable_logger

        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--models_list",
                        type = str,
                        required = True,
                        help = "Model names separated by comma. e.g. model1, model2, modeln")

    parser.add_argument("--max_tokens",
                        type = int,
                        required = False,
                        help = "Max new tokens. Default: 32")

    parser.add_argument("--max_test_instances",
                        type = int,
                        required = False,
                        help = "Max test intances to be inferred otherwise full dataset.")
    
    parser.add_argument("--watsonx_model",
                        type = bool,
                        required = False,
                        default=False,
                        help = "Indicates if it is a model hosted at WatsonX.ai.")

    parser.add_argument("--overwrite",
                        type = bool,
                        required = False,
                        default=False,
                        help = "Run evaluation again (using predicitions files if any) and save results.")

    args = parser.parse_args()

    main(args)
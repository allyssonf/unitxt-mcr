import logging
import argparse
import time
import datetime
import os

from evaluations.mmlu import Mmlu
from interfaces.evaluation import Evaluation
from utils.files import create_path
from dotenv import load_dotenv

load_dotenv(dotenv_path=f'{os.getcwd()}/meta-cascade')

logger = logging.getLogger(__name__)

def main(args):
    if not args.watsonx_model:
        if not os.getenv("CUDA_VISIBLE_DEVICES"):
            raise Exception("CUDA_VISIBLE_DEVICES env variable is needed to run local inference!")

    models_list = args.models_list.split(',')

    for model_name in models_list:
        new_evaluation = Mmlu(model_name, args)

        pretty_name = new_evaluation.get_pretty_name()

        create_path(f'{os.getcwd()}/logs', ignore_home=True)

        logging.basicConfig(
            filename=f"./logs/{int(time.time())}_{pretty_name}_execution.log", level=logging.INFO
        )

        if not issubclass(type(new_evaluation), Evaluation):
            raise Exception("Subclass of Evaluation interface expected!")

        start_time = time.perf_counter()

        logger.info(f"[MODEL START] ({pretty_name}): {start_time}")

        try:
            # Exceptions might arise while processing model's evaluation
            new_evaluation.evaluate()
        except Exception as error:
            logger.info(error)
            logger.info(f'Error evaluating model {new_evaluation.get_pretty_name()}')

        end_time=time.perf_counter()

        logger.info(f"[MODEL END] ({pretty_name}): {end_time}")

        execution_time=str(datetime.timedelta(seconds=end_time-start_time))

        logger.info(f"[EXECUTION TIME] ({pretty_name}): {execution_time}")

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

    args = parser.parse_args()

    main(args)
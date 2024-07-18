import logging
import argparse
from evaluations.mmlu import Mmlu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main(args):
    logger.info('Starting evaluation!')

    new_evaluation = Mmlu(args.model_name, 
                          args.max_tokens,
                          args.max_test_instances)

    new_evaluation.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name",
                        type = str,
                        required = True,
                        help = "Model name")
    parser.add_argument("--max_tokens",
                        type = int,
                        required = False,
                        help = "Max new tokens. Default: 32")

    parser.add_argument("--max_test_instances",
                    type = int,
                    required = False,
                    help = "Max test intances to be inferred otherwise full dataset.")

    args = parser.parse_args()

    main(args)
import logging
from evaluations.mmlu_llma_llama_3_8b_instruct import MMLU_Llama3_8b_Instruct

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    logger.info('Starting evaluation!')
    new_evaluation = MMLU_Llama3_8b_Instruct()
    new_evaluation.evaluate()

if __name__ == "__main__":
    main()
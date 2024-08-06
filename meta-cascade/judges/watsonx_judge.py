import json
import os

from interfaces.judge import Judge
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import get_model_specs
from pydantic import BaseModel, ConfigDict
from prompts.prompts import generate_new_prompt


# Retrieved from get_model_specs
wx_available_models = [
    'bigscience/mt0-xxl',
    'codellama/codellama-34b-instruct-hf',
    'google/flan-t5-xl',
    'google/flan-t5-xxl',
    'google/flan-ul2',
    'ibm-mistralai/merlinite-7b',
    'ibm/granite-13b-chat-v2',
    'ibm/granite-13b-instruct-v2',
    'ibm/granite-20b-code-instruct',
    'ibm/granite-20b-multilingual',
    'ibm/granite-34b-code-instruct',
    'ibm/granite-3b-code-instruct',
    'ibm/granite-7b-lab',
    'ibm/granite-8b-code-instruct',
    'ibm/slate-125m-english-rtrvr',
    'ibm/slate-30m-english-rtrvr',
    'intfloat/multilingual-e5-large',
    'meta-llama/llama-2-13b-chat',
    'meta-llama/llama-2-70b-chat',
    'meta-llama/llama-3-1-8b-instruct',
    'meta-llama/llama-3-405b-instruct',
    'meta-llama/llama-3-70b-instruct',
    'meta-llama/llama-3-8b-instruct',
    'mistralai/mistral-large',
    'mistralai/mixtral-8x7b-instruct-v01',
    'sentence-transformers/all-minilm-l12-v2'
]


class ExtendedBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_config["protected_namespaces"] = ()


class JudgesReponse(ExtendedBaseModel):
    correctness_score: float
    justificaiton: str


class WatsonXAIJudge(Judge):
    wx_client: APIClient
    wx_credentials: Credentials
    wx_model_name: str

    def __init__(self, model_name: str) -> None:
        self.wx_credentials = Credentials(
            url=os.getenv("WXAI_URL"),
            api_key=os.getenv("WXAI_API_KEY")
        )

        self.wx_client = APIClient(self.wx_credentials)

        self.wx_model_name = model_name

        wx_models = [model["model_id"] for model in get_model_specs(self.wx_credentials.url)["resources"]]

        if self.wx_model_name not in wx_models:
            raise Exception("Model not available in WatsonX.ai!")

    def infer(self):
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        parameters = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 256,
            GenParams.STOP_SEQUENCES: ["\n\n"]
        }

        from ibm_watsonx_ai.foundation_models import ModelInference

        model = ModelInference(
            model_id=self.wx_model_name,
            api_client=self.wx_client,
            params=parameters,
            project_id=os.getenv("WXAI_PROJECT_ID")
        )

        # Sample data
        question = "What's the process through which trees convert sunlight to energy that they can use as food?"
        model_answer = "Chromatology"
        golden_answer = "Photosynthesis"

        prompt = generate_new_prompt(question, model_answer, golden_answer)

        print('Generating text...\n')

        result = model.generate_text(prompt)

        llmaj: JudgesReponse = {}

        try:
            as_json = json.loads(result)

            llmaj = JudgesReponse(
                correctness_score=as_json['correctness_score'],
                justificaiton=as_json['justification']
            )
        except Exception as error:
            print(error)
            llmaj = JudgesReponse(
                correctness_score=-1.0,
                justificaiton=result
            )

        print(llmaj.model_dump())

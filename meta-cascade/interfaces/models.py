import gc
import torch

from unitxt.inference import HFPipelineBasedInferenceEngine, IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParams

class ModelLoader():
    _instance: None = None
    inference_model: HFPipelineBasedInferenceEngine | IbmGenAiInferenceEngine | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)

        return cls._instance

    def get_inference_model(self, model_name: str, watsonx: bool, max_tokens: int = 32) -> HFPipelineBasedInferenceEngine | IbmGenAiInferenceEngine:
        if self.inference_model is None:
            if not watsonx:
                self.inference_model = HFPipelineBasedInferenceEngine(
                    model_name=model_name, max_new_tokens=max_tokens
                )
            else:
                gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=32)

                self.inference_model = IbmGenAiInferenceEngine(
                    model_name=model_name, parameters=gen_params
                )

        return self.inference_model
    
    def new_inference_model(self, model_name: str, watsonx: bool, max_tokens: int = 32) -> HFPipelineBasedInferenceEngine | IbmGenAiInferenceEngine:
        del self.inference_model

        gc.collect()
        torch.cuda.empty_cache()

        return self.get_inference_model(model_name, watsonx, max_tokens)

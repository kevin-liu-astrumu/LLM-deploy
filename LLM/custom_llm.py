import json
from typing import Optional, List, Mapping, Any
import requests
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback


class OurLLM(CustomLLM):

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted under.",
    )
    model: str = Field(description="The Ollama model to use.")
    headers: Dict[str, Any] =Field(
        default= {"Content-Type": "application/json"}, 
        description = '')


    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Your are a helpful AI assist."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0,
            "stop": ['.'],
        }

        response = requests.post(self.base_url, headers=self.headers, data=json.dumps(data))
        content = response.json()["choices"][0]["message"]["content"]
        return CompletionResponse(text=content)
    
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in "dummy response":
            response += token
            yield CompletionResponse(text=response, delta=token)
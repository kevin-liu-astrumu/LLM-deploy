from typing import Any, Dict
from openai import OpenAI

from llama_index.core.llms import LLMMetadata
from llama_index.legacy.bridge.pydantic import Field
from llama_index.legacy.core.llms.types import CompletionResponse, CompletionResponseGen
from llama_index.legacy.llms.base import llm_completion_callback
from llama_index.legacy.llms.custom import CustomLLM


class LMStudio_LLM(CustomLLM):
    base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base url the model is hosted under.",
    )
    model: str = Field(
        default="local-model",
        description="Model used for generation.")
    system_prompt: str = Field(
        default="You are a helpful AI assistant. Your task is to help user with their requests.",
        description="The system prompt used for LLM generation.",
    )
    temperature: float = Field(
        default=0,
        description="The temperature to use for generation.",
    )
    stop: list = Field(
        default=["."], description="Stop token used for stoping LLM generation."
    )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {"temperature": self.temperature, "stop": self.stop}
        return {
            **base_kwargs,
        }

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        client = OpenAI(base_url=self.base_url, api_key="not-needed")
        completion = client.chat.completions.create(
            model=self.model, # this field is currently unused
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        text = completion.choices[0].message.content
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        client = OpenAI(base_url=self.base_url, api_key="not-needed")
        completion = client.chat.completions.create(
            model=self.model, # this field is currently unused
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        response = ""
        text = completion.choices[0].message.content
        for token in text:
            response += token 
            yield CompletionResponse(text=text, delta=token)


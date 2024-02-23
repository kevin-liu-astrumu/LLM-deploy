import json
from typing import Any, Dict

import httpx
from httpx import Timeout
from llama_index.core.llms import LLMMetadata
from llama_index.legacy.bridge.pydantic import Field
from llama_index.legacy.core.llms.types import CompletionResponse, CompletionResponseGen
from llama_index.legacy.llms.base import llm_completion_callback
from llama_index.legacy.llms.custom import CustomLLM

DEFAULT_REQUEST_TIMEOUT = 60.0


class Custom_LLM(CustomLLM):
    base_url: str = Field(
        description="Base url the model is hosted under.",
    )
    model: str = Field(description="Model used for generation.")
    system_prompt: str = Field(
        default="You are a helpful AI assistant. Your task is to help user with their requests.",
        description="The system prompt used for LLM generation.",
    )
    temperature: float = Field(
        default=0,
        description="The temperature to use for generation.",
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to Ollama API server",
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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Your are a helpful AI assist."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "stop": self.stop,
            "stream": False,
        }
        with httpx.Client(timeout=Timeout(DEFAULT_REQUEST_TIMEOUT)) as client:
            response = client.post(
                url=f"{self.base_url}",
                json=payload,
            )
            raw = response.json()
            text = raw.get("choices")[0].get("message").get("content")

            return CompletionResponse(text=text, raw=raw)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Your are a helpful AI assist."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "stop": self.stop,
            "stream": False,
        }
        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            with client.stream(
                method="POST",
                url=f"{self.base_url}",
                json=payload,
            ) as response:
                response.raise_for_status()
                text = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        delta = chunk.get("choices")[0].get("message").get("content")
                        text += delta
                        yield CompletionResponse(
                            delta=delta,
                            text=text,
                            raw=chunk,
                        )

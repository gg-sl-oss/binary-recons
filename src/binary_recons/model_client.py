"""Instructor-backed, stage-specific structured requests to llama.cpp."""

from __future__ import annotations

from typing import Any

import instructor
from instructor.core.exceptions import InstructorRetryException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from .llama_server import ManagedLlamaServer
from .models import (
    ContractProposal,
    ModelPreset,
    SearchConfig,
    SimilarityPatch,
    SourcePatch,
    SymbolProposalBatch,
)
from .prompts import SYSTEM_PROMPT


class ModelRequestError(RuntimeError):
    """A bounded model request failed without producing validated output."""


class StructuredModelClient:
    """Expose only the four small decisions allowed by the search driver."""

    def __init__(self, server: ManagedLlamaServer, config: SearchConfig):
        self.server = server
        self.config = config
        self._openai = OpenAI(
            base_url=server.base_url,
            api_key="local-llama-server",
            timeout=config.request_timeout,
            max_retries=0,
        )
        self._instructor = instructor.from_openai(
            self._openai,
            mode=instructor.Mode.JSON_SCHEMA,
        )

    def close(self) -> None:
        self._openai.close()

    def infer_contract(self, prompt: str) -> tuple[ContractProposal, Any]:
        return self._request(
            "contract",
            prompt,
            ContractProposal,
            seed=self.config.seed,
            max_tokens=192,
        )

    def propose_symbols(
        self,
        prompt: str,
        attempt: int,
    ) -> tuple[SymbolProposalBatch, Any]:
        return self._request(
            "symbol repair",
            prompt,
            SymbolProposalBatch,
            seed=self.config.seed + 100 + attempt,
            max_tokens=160,
            diverse=True,
        )

    def repair_compile(
        self,
        prompt: str,
        round_number: int,
    ) -> tuple[SourcePatch, Any]:
        return self._request(
            "compile repair",
            prompt,
            SourcePatch,
            seed=self.config.seed + 1000 + round_number,
            max_tokens=320,
        )

    def improve_similarity(
        self,
        prompt: str,
        round_number: int,
    ) -> tuple[SimilarityPatch, Any]:
        return self._request(
            "similarity edit",
            prompt,
            SimilarityPatch,
            seed=self.config.seed + 2000 + round_number,
            max_tokens=192,
        )

    def _request(
        self,
        stage: str,
        prompt: str,
        response_model: type[BaseModel],
        seed: int,
        max_tokens: int,
        diverse: bool = False,
    ) -> tuple[Any, Any]:
        self.server.ensure_alive()
        preset = self.server.config.resolved_preset()
        extra_body = {
            "top_k": self.config.effective_top_k(preset),
            "min_p": self.config.min_p,
            "repeat_penalty": self.config.repeat_penalty,
            "cache_prompt": True,
        }
        if preset in (ModelPreset.QWEN, ModelPreset.GEMMA):
            extra_body["reasoning_effort"] = self.config.reasoning_effort
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.config.thinking,
            }
        elif self.config.thinking:
            extra_body["reasoning_effort"] = self.config.reasoning_effort

        temperature = self.config.effective_temperature(preset)
        presence_penalty = self.config.effective_presence_penalty(preset)
        if (
            preset == ModelPreset.QWEN
            and not self.config.thinking
            and self.config.temperature is None
            and not diverse
        ):
            # The bounded edit POC was materially more stable at low temperature.
            temperature = 0.2
        if (
            preset == ModelPreset.QWEN
            and not self.config.thinking
            and self.config.presence_penalty is None
            and not diverse
        ):
            presence_penalty = 0.0

        try:
            result, completion = (
                self._instructor.chat.completions.create_with_completion(
                    model=self.config.model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_retries=self.config.format_retries,
                    max_tokens=min(self.config.max_tokens, max_tokens),
                    temperature=temperature,
                    top_p=self.config.effective_top_p(preset),
                    presence_penalty=presence_penalty,
                    seed=seed,
                    extra_body=extra_body,
                )
            )
        except (InstructorRetryException, OpenAIError) as error:
            raise ModelRequestError("%s request failed: %s" % (stage, error)) from error
        self.server.ensure_alive()
        return result, completion

"""Instructor-backed structured candidate generation."""

from __future__ import annotations

from typing import Any

import instructor
from openai import OpenAI
from pydantic import BaseModel

from .llama_server import ManagedLlamaServer
from .models import (
    Candidate,
    CandidateBatch,
    ModelPreset,
    SearchConfig,
    SymbolProposalBatch,
)
from .prompts import SYSTEM_PROMPT


class CandidateGenerator:
    def __init__(
        self,
        server: ManagedLlamaServer,
        config: SearchConfig,
    ):
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

    def generate(self, prompt: str, iteration: int) -> tuple[CandidateBatch, Any]:
        return self._request(
            prompt,
            CandidateBatch,
            self.config.seed + iteration,
        )

    def repair(
        self,
        prompt: str,
        iteration: int,
        candidate_index: int,
        repair_attempt: int,
    ) -> tuple[Candidate, Any]:
        seed = (
            self.config.seed + iteration * 1000 + candidate_index * 10 + repair_attempt
        )
        return self._request(prompt, Candidate, seed)

    def propose_symbols(
        self,
        prompt: str,
        iteration: int,
        candidate_index: int,
        repair_attempt: int,
    ) -> tuple[SymbolProposalBatch, Any]:
        seed = (
            self.config.seed + iteration * 1000 + candidate_index * 10 + repair_attempt
        )
        return self._request(
            prompt,
            SymbolProposalBatch,
            seed,
            max_tokens=min(self.config.max_tokens, 256),
        )

    def _request(
        self,
        prompt: str,
        response_model: type[BaseModel],
        seed: int,
        max_tokens: int | None = None,
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
        batch, completion = self._instructor.chat.completions.create_with_completion(
            model=self.config.model,
            response_model=response_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_retries=self.config.format_retries,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.effective_temperature(preset),
            top_p=self.config.effective_top_p(preset),
            presence_penalty=self.config.effective_presence_penalty(preset),
            seed=seed,
            extra_body=extra_body,
        )
        self.server.ensure_alive()
        return batch, completion

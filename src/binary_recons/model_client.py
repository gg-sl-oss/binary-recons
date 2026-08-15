"""Instructor-backed structured candidate generation."""

from __future__ import annotations

from typing import Any

import instructor
from openai import OpenAI

from .llama_server import ManagedLlamaServer
from .models import Candidate, CandidateBatch, SearchConfig
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

    def _request(
        self,
        prompt: str,
        response_model: type[Candidate] | type[CandidateBatch],
        seed: int,
    ) -> tuple[Any, Any]:
        self.server.ensure_alive()
        extra_body = {
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "repeat_penalty": self.config.repeat_penalty,
            "cache_prompt": True,
            "reasoning_effort": self.config.reasoning_effort,
            "chat_template_kwargs": {
                "enable_thinking": self.config.thinking,
            },
        }
        batch, completion = self._instructor.chat.completions.create_with_completion(
            model=self.config.model,
            response_model=response_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_retries=self.config.format_retries,
            max_tokens=self.config.max_tokens,
            temperature=self.config.effective_temperature,
            top_p=self.config.effective_top_p,
            presence_penalty=self.config.effective_presence_penalty,
            seed=seed,
            extra_body=extra_body,
        )
        self.server.ensure_alive()
        return batch, completion

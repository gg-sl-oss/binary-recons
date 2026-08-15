"""Typed configuration and model-output contracts."""

from __future__ import annotations

import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MODEL_ENVIRONMENT_VARIABLE = "BINARY_RECONS_MODEL_PATH"


def discover_default_model_path() -> Path | None:
    configured = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    return Path(configured).expanduser() if configured else None


DEFAULT_MODEL_PATH = discover_default_model_path()

MAX_CANDIDATE_CHARS = 12000


class ServerMode(str, Enum):
    MANAGED = "managed"
    EXTERNAL = "external"


class ModelPreset(str, Enum):
    AUTO = "auto"
    GENERIC = "generic"
    QWEN = "qwen"
    GEMMA = "gemma"


class Candidate(BaseModel):
    """One complete source definition proposed by the model."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        min_length=20,
        max_length=MAX_CANDIDATE_CHARS,
        description=(
            "Complete target definition, including its Function start marker; "
            "no Markdown or explanation."
        ),
    )

    @field_validator("source")
    @classmethod
    def strip_source(cls, value: str) -> str:
        return value.strip()


class CandidateBatch(BaseModel):
    """A bounded set of source hypotheses generated in one request."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[Candidate] = Field(min_length=1, max_length=8)


class TargetSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    address: int
    symbol: str
    source_path: Path
    prototype: str
    assembly_path: Path
    decompiled_path: Path

    @property
    def source_display(self) -> str:
        try:
            return str(self.source_path.relative_to(self.root))
        except ValueError:
            return str(self.source_path)


class EvidenceBundle(BaseModel):
    language: str
    compiler: str
    project_guidance: str
    original_assembly: str
    decompiler_hint: str
    callee_evidence: str
    declaration_evidence: str


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "local-model"
    max_iterations: int = Field(default=3, ge=1, le=20)
    candidates_per_iteration: int = Field(default=3, ge=1, le=8)
    target_score: float = Field(default=80.0, ge=0.0, le=100.0)
    max_tokens: int = Field(default=1200, ge=128)
    reasoning_effort: Literal["none", "low", "medium"] = "none"
    history_limit: int = Field(default=4, ge=0, le=20)
    max_callees: int = Field(default=8, ge=0, le=32)
    request_timeout: float = Field(default=180.0, gt=0)
    build_timeout: float = Field(default=120.0, gt=0)
    format_retries: int = Field(default=1, ge=0, le=3)
    compile_repair_attempts: int = Field(default=1, ge=0, le=3)
    seed: int
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repeat_penalty: float = Field(default=1.0, gt=0.0)

    @property
    def thinking(self) -> bool:
        return self.reasoning_effort != "none"

    def effective_temperature(self, preset: ModelPreset) -> float:
        if self.temperature is not None:
            return self.temperature
        if preset == ModelPreset.GEMMA:
            return 1.0
        return 1.0 if self.thinking else 0.7

    def effective_top_p(self, preset: ModelPreset) -> float:
        if self.top_p is not None:
            return self.top_p
        if preset == ModelPreset.GEMMA:
            return 0.95
        return 0.95 if self.thinking else 0.80

    def effective_top_k(self, preset: ModelPreset) -> int:
        if self.top_k is not None:
            return self.top_k
        if preset == ModelPreset.GEMMA:
            return 64
        if preset == ModelPreset.QWEN:
            return 20
        return 40

    def effective_presence_penalty(self, preset: ModelPreset) -> float:
        if self.presence_penalty is not None:
            return self.presence_penalty
        if preset == ModelPreset.GEMMA:
            return 0.0
        if preset == ModelPreset.QWEN:
            return 0.0 if self.thinking else 1.5
        return 0.0


class LlamaServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    mode: ServerMode = ServerMode.MANAGED
    binary: Path | None = None
    model_path: Path | None = DEFAULT_MODEL_PATH
    alias: str = "local-model"
    preset: ModelPreset = ModelPreset.AUTO
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    context_size: int = Field(default=16384, ge=2048)
    startup_timeout: float = Field(default=240.0, gt=0)
    shutdown_timeout: float = Field(default=15.0, gt=0)
    health_interval: float = Field(default=0.5, gt=0)
    command_override: list[str] | None = Field(default=None, exclude=True)

    @property
    def base_url(self) -> str:
        return "http://%s:%d/v1" % (self.host, self.port)

    @property
    def health_url(self) -> str:
        return "http://%s:%d/health" % (self.host, self.port)

    def resolved_binary(self) -> Path:
        if self.binary is not None:
            return self.binary
        discovered = shutil.which("llama-server")
        if discovered is None:
            raise RuntimeError(
                "llama-server was not found; pass --llama-bin explicitly"
            )
        return Path(discovered)

    def resolved_preset(self) -> ModelPreset:
        if self.preset != ModelPreset.AUTO:
            return self.preset
        identity = self.alias
        if self.model_path is not None:
            identity += " " + self.model_path.name
        identity = identity.lower()
        if "qwen" in identity:
            return ModelPreset.QWEN
        if "gemma" in identity:
            return ModelPreset.GEMMA
        return ModelPreset.GENERIC

    def command(self, search: SearchConfig) -> list[str]:
        if self.command_override is not None:
            return list(self.command_override)
        if self.model_path is None:
            raise RuntimeError(
                "no model configured; pass --model-path or set BINARY_RECONS_MODEL_PATH"
            )
        preset = self.resolved_preset()
        command = [
            str(self.resolved_binary()),
            "-m",
            str(self.model_path),
            "--alias",
            self.alias,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            "1",
            "--batch-size",
            "256",
            "--ubatch-size",
            "64",
            "--n-gpu-layers",
            "all",
            "--flash-attn",
            "on",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--temp",
            str(search.effective_temperature(preset)),
            "--top-p",
            str(search.effective_top_p(preset)),
            "--top-k",
            str(search.effective_top_k(preset)),
            "--min-p",
            str(search.min_p),
            "--presence-penalty",
            str(search.effective_presence_penalty(preset)),
            "--repeat-penalty",
            str(search.repeat_penalty),
            "--jinja",
            "--metrics",
        ]
        if preset == ModelPreset.QWEN:
            command.extend(
                [
                    "--spec-type",
                    "draft-mtp",
                    "--spec-draft-n-max",
                    "2",
                    "--spec-draft-ngl",
                    "all",
                    "--spec-draft-type-k",
                    "q4_0",
                    "--spec-draft-type-v",
                    "q4_0",
                ]
            )
        return command

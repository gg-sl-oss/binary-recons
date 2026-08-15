"""Typed configuration and model-output contracts."""

from __future__ import annotations

import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MODEL_ENVIRONMENT_VARIABLE = "BINARY_RECONS_MODEL_PATH"
MODEL_FILENAME = "Qwen3.8-27B-BF16-00001-of-00002.gguf"


def discover_default_model_path() -> Path | None:
    configured = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser()
    snapshots = (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--unsloth--Qwen3.8-27B-GGUF/snapshots"
    )
    candidates = sorted(snapshots.glob("*/BF16/%s" % MODEL_FILENAME))
    return candidates[-1] if candidates else None


DEFAULT_MODEL_PATH = discover_default_model_path()

MAX_CANDIDATE_CHARS = 12000


class ServerMode(str, Enum):
    MANAGED = "managed"
    EXTERNAL = "external"


class Candidate(BaseModel):
    """One complete C definition proposed by the model."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        min_length=20,
        max_length=MAX_CANDIDATE_CHARS,
        description=(
            "Complete target C definition, including its Function start marker; "
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
    original_assembly: str
    decompiler_hint: str
    callee_evidence: str
    declaration_evidence: str


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "qwen3.8-27b"
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
    seed: int
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repeat_penalty: float = Field(default=1.0, gt=0.0)

    @property
    def thinking(self) -> bool:
        return self.reasoning_effort != "none"

    @property
    def effective_temperature(self) -> float:
        if self.temperature is not None:
            return self.temperature
        return 1.0 if self.thinking else 0.7

    @property
    def effective_top_p(self) -> float:
        if self.top_p is not None:
            return self.top_p
        return 0.95 if self.thinking else 0.80

    @property
    def effective_presence_penalty(self) -> float:
        if self.presence_penalty is not None:
            return self.presence_penalty
        return 0.0 if self.thinking else 1.5


class LlamaServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    mode: ServerMode = ServerMode.MANAGED
    binary: Path | None = None
    model_path: Path | None = DEFAULT_MODEL_PATH
    alias: str = "qwen3.8-27b"
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

    def command(self, search: SearchConfig) -> list[str]:
        if self.command_override is not None:
            return list(self.command_override)
        if self.model_path is None:
            raise RuntimeError(
                "no Qwen model found; pass --model-path or set BINARY_RECONS_MODEL_PATH"
            )
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
            "--temp",
            str(search.effective_temperature),
            "--top-p",
            str(search.effective_top_p),
            "--top-k",
            str(search.top_k),
            "--min-p",
            str(search.min_p),
            "--presence-penalty",
            str(search.effective_presence_penalty),
            "--repeat-penalty",
            str(search.repeat_penalty),
            "--jinja",
            "--metrics",
        ]
        return command

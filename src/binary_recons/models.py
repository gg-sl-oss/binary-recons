"""Typed configuration and model-output contracts."""

from __future__ import annotations

import os
import re
import shutil
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MODEL_ENVIRONMENT_VARIABLE = "BINARY_RECONS_MODEL_PATH"
DEFAULT_HUGGINGFACE_HUB_CACHE = Path("~/.cache/huggingface/hub")
_SHARDED_GGUF_RE = re.compile(
    r"-(?P<shard>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.I,
)
_AUXILIARY_GGUF_NAMES = ("mmproj", "tokenizer", "vision")


def _cached_model_rank(path: Path) -> tuple[tuple[int, ...], float, int, float, str]:
    """Prefer newer, larger, higher-fidelity cached Qwen model variants."""

    identity = str(path).lower()
    version_match = re.search(r"qwen[-_]?([0-9]+(?:\.[0-9]+)*)", identity)
    version = (
        tuple(int(part) for part in version_match.group(1).split("."))
        if version_match is not None
        else ()
    )
    size_match = re.search(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)b\b", identity)
    size = float(size_match.group(1)) if size_match is not None else 0.0
    fidelity = next(
        (
            rank
            for label, rank in (
                ("bf16", 9),
                ("f16", 8),
                ("q8", 7),
                ("q6", 6),
                ("q5", 5),
                ("q4", 4),
                ("q3", 3),
                ("q2", 2),
            )
            if label in identity
        ),
        0,
    )
    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = 0.0
    return version, size, fidelity, modified, str(path)


def _has_all_model_shards(path: Path, shard: re.Match[str]) -> bool:
    """Reject partially downloaded split GGUFs before llama.cpp sees them."""

    shard_count = int(shard.group("count"))
    start, end = shard.span("shard")
    return all(
        path.with_name(path.name[:start] + f"{index:05d}" + path.name[end:]).is_file()
        for index in range(1, shard_count + 1)
    )


def discover_cached_qwen_model(cache_root: Path) -> Path | None:
    """Select a loadable first GGUF shard from a Hugging Face hub cache."""

    cache_root = cache_root.expanduser()
    if not cache_root.is_dir():
        return None
    candidates: list[Path] = []
    for path in cache_root.glob("models--*/snapshots/**/*.gguf"):
        identity = str(path).lower()
        if "qwen" not in identity or any(
            excluded in path.name.lower() for excluded in _AUXILIARY_GGUF_NAMES
        ):
            continue
        shard = _SHARDED_GGUF_RE.search(path.name)
        if shard is not None:
            if int(shard.group("shard")) != 1 or not _has_all_model_shards(path, shard):
                continue
        if path.is_file():
            candidates.append(path)
    return max(candidates, key=_cached_model_rank) if candidates else None


def discover_default_model_path(cache_root: Path | None = None) -> Path | None:
    """Resolve an explicit model override, then a cached Qwen GGUF."""

    configured = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser()
    return discover_cached_qwen_model(
        cache_root or DEFAULT_HUGGINGFACE_HUB_CACHE,
    )


DEFAULT_MODEL_PATH = discover_default_model_path()

MAX_CANDIDATE_CHARS = 32000
MAX_SUPPORTING_INSERTION_CHARS = 8000
MAX_SUPPORTING_TOTAL_CHARS = 24000
MAX_EXACT_EDIT_CHARS = 600


def _clean_multiline_text(value: str) -> str:
    """Strip model-added line-end whitespace before content reaches the workspace."""

    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def _split_prototype_parameters(prototype: str, symbol: str) -> list[str]:
    match = re.search(r"\b%s\s*\(" % re.escape(symbol), prototype)
    if match is None:
        return []
    start = prototype.find("(", match.start())
    depth = 0
    beginning = start + 1
    parameters: list[str] = []
    for index in range(beginning, len(prototype)):
        character = prototype[index]
        if character in "([":
            depth += 1
        elif character == "]":
            depth = max(0, depth - 1)
        elif character == ")":
            if depth == 0:
                parameters.append(prototype[beginning:index].strip())
                return parameters
            depth -= 1
        elif character == "," and depth == 0:
            parameters.append(prototype[beginning:index].strip())
            beginning = index + 1
    return []


def _parameter_has_name(parameter: str) -> bool:
    if parameter in ("", "void", "..."):
        return True
    if re.search(r"\(\s*\*\s*[A-Za-z_]\w*\s*\)", parameter):
        return True
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", parameter)
    keywords = {
        "char",
        "const",
        "double",
        "enum",
        "float",
        "int",
        "long",
        "register",
        "short",
        "signed",
        "struct",
        "union",
        "unsigned",
        "void",
        "volatile",
    }
    non_keywords = [
        identifier for identifier in identifiers if identifier not in keywords
    ]
    if not non_keywords:
        return False
    if len(non_keywords) >= 2:
        return True
    builtin_types = {
        "char",
        "double",
        "float",
        "int",
        "long",
        "short",
        "signed",
        "unsigned",
        "void",
    }
    return any(identifier in builtin_types for identifier in identifiers[:-1])


class ServerMode(str, Enum):
    MANAGED = "managed"
    EXTERNAL = "external"


class ModelPreset(str, Enum):
    AUTO = "auto"
    GENERIC = "generic"
    QWEN = "qwen"
    GEMMA = "gemma"


class SupportingInsertion(BaseModel):
    """One append-only declaration/definition block for a configured file."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        max_length=500,
        description="Exact configured project-relative support-file path.",
    )
    content: str = Field(
        min_length=1,
        max_length=MAX_SUPPORTING_INSERTION_CHARS,
        description=(
            "Complete new declarations or definitions to insert into the file; "
            "no surrounding file text or Markdown."
        ),
    )

    @field_validator("path")
    @classmethod
    def strip_path(cls, value: str) -> str:
        return value.strip()

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        return _clean_multiline_text(value)


class Candidate(BaseModel):
    """One complete, bounded source transaction evaluated by the driver."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Descriptive source-level function name without an address.",
    )
    prototype: str = Field(
        min_length=6,
        max_length=1000,
        description="Complete function signature without a trailing semicolon.",
    )
    source: str = Field(
        min_length=20,
        max_length=MAX_CANDIDATE_CHARS,
        description=(
            "Complete target definition, including its Function start marker; "
            "no Markdown or explanation."
        ),
    )
    supporting_insertions: list[SupportingInsertion] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "New target-required declarations/definitions for configured support "
            "files. Empty when the target needs none."
        ),
    )

    @field_validator("source")
    @classmethod
    def strip_source(cls, value: str) -> str:
        return _clean_multiline_text(value)

    @field_validator("prototype")
    @classmethod
    def strip_prototype(cls, value: str) -> str:
        return value.strip().rstrip(";").strip()

    @model_validator(mode="after")
    def contract_agrees(self) -> "Candidate":
        if re.search(r"\b%s\s*\(" % re.escape(self.symbol), self.prototype) is None:
            raise ValueError("prototype does not contain the proposed symbol")
        return self


class ContractProposal(BaseModel):
    """The small name-and-interface decision made before body integration."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Meaningful source-level function name without an address.",
    )
    prototype: str = Field(
        min_length=6,
        max_length=1000,
        description=(
            "Complete function signature, including parameter names, without a "
            "trailing semicolon."
        ),
    )

    @field_validator("prototype")
    @classmethod
    def strip_prototype(cls, value: str) -> str:
        return value.strip().rstrip(";").strip()

    @model_validator(mode="after")
    def contract_agrees(self) -> "ContractProposal":
        if re.search(r"\b%s\s*\(" % re.escape(self.symbol), self.prototype) is None:
            raise ValueError("prototype does not contain the proposed symbol")
        unnamed = [
            parameter
            for parameter in _split_prototype_parameters(
                self.prototype,
                self.symbol,
            )
            if not _parameter_has_name(parameter)
        ]
        if unnamed:
            raise ValueError(
                "every non-void contract parameter must have a name: %s"
                % ", ".join(unnamed)
            )
        return self


class ExactEdit(BaseModel):
    """One bounded textual substitution against the current working source."""

    model_config = ConfigDict(extra="forbid")

    old: str = Field(min_length=1, max_length=MAX_EXACT_EDIT_CHARS)
    new: str = Field(max_length=MAX_EXACT_EDIT_CHARS)
    mode: Literal["once", "all"] = "once"


class IdentifierReplacement(BaseModel):
    """A whole-token rename used mainly for unresolved decompiler identifiers."""

    model_config = ConfigDict(extra="forbid")

    old: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    new: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )


class SourcePatch(BaseModel):
    """A compile-repair response small enough to validate mechanically."""

    model_config = ConfigDict(extra="forbid")

    identifier_replacements: list[IdentifierReplacement] = Field(
        default_factory=list,
        max_length=8,
    )
    edits: list[ExactEdit] = Field(default_factory=list, max_length=1)
    supporting_insertions: list[SupportingInsertion] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "New declarations or definitions needed by a repaired source-level "
            "global. Use only configured support-file paths."
        ),
    )


class SimilarityPatch(BaseModel):
    """A single source-form experiment for an already-compiling function."""

    model_config = ConfigDict(extra="forbid")

    edit: ExactEdit


class SymbolProposalBatch(BaseModel):
    """Diverse model-proposed names for a candidate with a colliding symbol."""

    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(
        min_length=6,
        max_length=6,
        description=(
            "Distinct, meaningful C identifiers ordered from strongest to weakest."
        ),
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{2,99}", value):
                raise ValueError("every proposed symbol must be a valid C identifier")
            if value in cleaned:
                raise ValueError("proposed symbols must be distinct")
            cleaned.append(value)
        return cleaned


class TargetSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    address: int
    symbol: str | None
    source_path: Path
    prototype: str | None
    assembly_path: Path
    decompiled_path: Path

    @property
    def source_display(self) -> str:
        try:
            return str(self.source_path.relative_to(self.root))
        except ValueError:
            return str(self.source_path)

    @property
    def has_contract(self) -> bool:
        return self.symbol is not None and self.prototype is not None

    def with_candidate_contract(self, candidate: Candidate) -> "TargetSpec":
        return self.model_copy(
            update={"symbol": candidate.symbol, "prototype": candidate.prototype}
        )


class EvidenceBundle(BaseModel):
    language: str
    compiler: str
    project_guidance: str
    original_assembly: str
    decompiler_hint: str
    string_evidence: str
    reserved_symbols: str
    callee_evidence: str
    declaration_evidence: str
    supporting_file_evidence: str


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "local-model"
    max_edits: int = Field(default=4, ge=0, le=20)
    target_score: float = Field(default=80.0, ge=0.0, le=100.0)
    max_tokens: int = Field(default=768, ge=64, le=2000)
    reasoning_effort: Literal["none", "low", "medium"] = "none"
    max_callees: int = Field(default=2, ge=0, le=32)
    request_timeout: float = Field(default=60.0, gt=0)
    build_timeout: float = Field(default=120.0, gt=0)
    format_retries: int = Field(default=1, ge=0, le=3)
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
    context_size: int = Field(default=32768, ge=2048)
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
                "no model configured; pass --model-path, set "
                "BINARY_RECONS_MODEL_PATH, or cache a Qwen GGUF under "
                "~/.cache/huggingface/hub"
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

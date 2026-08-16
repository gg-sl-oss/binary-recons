"""Target-project configuration with no game-specific conventions."""

from __future__ import annotations

import re
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_CONFIG_NAME = "binary-recons.toml"
PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def load_rule_profile(name: str) -> str:
    """Load one packaged, reusable rule profile by its stable name."""

    if PROFILE_NAME_RE.fullmatch(name) is None:
        raise RuntimeError("invalid rule profile name: %s" % name)
    directory = files("binary_recons").joinpath("profiles")
    resource = directory.joinpath(name + ".txt")
    if not resource.is_file():
        available = sorted(
            path.name.removesuffix(".txt") for path in directory.iterdir()
        )
        raise RuntimeError(
            "unknown rule profile %r; available profiles: %s"
            % (name, ", ".join(available) or "none")
        )
    return resource.read_text(encoding="utf-8").strip()


class SourceUnit(BaseModel):
    """One source file's inclusive original-address range."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "SourceUnit":
        if self.end < self.start:
            raise ValueError("source-unit end must be at or after its start")
        return self


class SupportFile(BaseModel):
    """One explicitly writable file for model-proposed supporting insertions."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    purpose: str = Field(min_length=1, max_length=500)
    insertion: Literal["auto", "append", "before-final-endif"] = "auto"


class ProjectConfig(BaseModel):
    """All target layout, compiler, prompt, and comparison conventions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    language: str = Field(min_length=1)
    compiler: str = Field(min_length=1)
    exports_dir: Path = Path("ghidra")
    output_dir: Path = Path("out/binary-recons")
    strings_file: Path | None = None
    source_dirs: list[Path] = Field(default_factory=lambda: [Path("src")])
    declaration_files: list[str] = Field(default_factory=list)
    prototype_file: Path | None = None
    rule_profiles: list[str] = Field(default_factory=list)
    prompt_files: list[Path] = Field(default_factory=list)
    support_files: list[SupportFile] = Field(default_factory=list)
    compare_command: list[str] = Field(min_length=1)
    source_units: list[SourceUnit] = Field(default_factory=list)
    skip_addresses: list[int] = Field(default_factory=list)
    require_global_address_suffix: bool = False

    @model_validator(mode="after")
    def validate_support_files(self) -> "ProjectConfig":
        paths = [support.path for support in self.support_files]
        if len(paths) != len(set(paths)):
            raise ValueError("support-file paths must be unique")
        if self.prototype_file is not None and self.prototype_file in paths:
            raise ValueError(
                "prototype_file is managed separately and cannot be a support file"
            )
        if len(self.skip_addresses) != len(set(self.skip_addresses)):
            raise ValueError("skip_addresses must be unique")
        if any(address < 0 for address in self.skip_addresses):
            raise ValueError("skip_addresses cannot contain negative addresses")
        return self

    def resolve(self, root: Path, path: Path) -> Path:
        return path if path.is_absolute() else root / path

    def source_paths(self, root: Path) -> list[Path]:
        return [self.resolve(root, path) for path in self.source_dirs]

    def declarations(self, root: Path) -> list[Path]:
        paths: list[Path] = []
        for pattern in self.declaration_files:
            candidate = Path(pattern)
            if candidate.is_absolute():
                matches = [candidate] if candidate.is_file() else []
            else:
                matches = sorted(path for path in root.glob(pattern) if path.is_file())
            for match in matches:
                resolved = match.resolve()
                if resolved not in paths:
                    paths.append(resolved)
        return paths

    def guidance(self, root: Path, character_limit: int = 16000) -> str:
        blocks: list[str] = []
        if self.require_global_address_suffix:
            blocks.append(
                "[configured global naming policy]\n"
                "- Every newly declared source-level global must end with its "
                "evidenced hexadecimal address, for example "
                "g_windowHandle_00412000."
            )
        for profile in self.rule_profiles:
            blocks.append(
                "[shared profile: %s]\n%s" % (profile, load_rule_profile(profile))
            )
        for configured in self.prompt_files:
            path = self.resolve(root, configured)
            if not path.exists():
                raise RuntimeError("missing project prompt file: %s" % path)
            blocks.append(
                "[project file: %s]\n%s"
                % (configured, path.read_text(encoding="utf-8"))
            )
        text = "\n\n".join(blocks).strip()
        if not text:
            return "No additional target-project rules were supplied."
        if len(text) <= character_limit:
            return text
        return text[:character_limit] + "\n... project guidance truncated ..."

    def comparison_command(self, symbol: str, address: int) -> list[str]:
        values = {
            "symbol": symbol,
            "address": address,
            "address_hex": "%08X" % address,
        }
        try:
            return [argument.format(**values) for argument in self.compare_command]
        except (KeyError, ValueError) as error:
            raise RuntimeError(
                "invalid compare_command placeholder: %s" % error
            ) from error


def load_project_config(root: Path, configured_path: Path | None) -> ProjectConfig:
    path = configured_path or Path(DEFAULT_CONFIG_NAME)
    path = path if path.is_absolute() else root / path
    if not path.exists():
        raise RuntimeError(
            "missing project configuration: %s; create %s or pass --config"
            % (path, DEFAULT_CONFIG_NAME)
        )
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return ProjectConfig.model_validate(data)

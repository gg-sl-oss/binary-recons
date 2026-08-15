"""File-backed run logging."""

from __future__ import annotations

import re
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .models import (
    Candidate,
    CandidateBatch,
    SearchConfig,
    SymbolProposalBatch,
    TargetSpec,
)
from .utils import atomic_write, utc_now, write_json


class RunLog:
    def __init__(self, output_root: Path, target: TargetSpec, config: SearchConfig):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        model_name = re.sub(r"[^A-Za-z0-9._-]+", "-", config.model).strip("-.")
        model_name = model_name[:80] or "local-model"
        self.directory = output_root / model_name / ("%08X" % target.address) / stamp
        self.directory.mkdir(parents=True, exist_ok=False)
        self._started_monotonic = time.monotonic()
        self._manifest: dict[str, Any] = {
            "started_at": utc_now(),
            "status": "running",
            "target": target.model_dump(mode="json"),
            "search": config.model_dump(mode="json"),
            "python_dependencies": {
                package: self._package_version(package)
                for package in ("instructor", "openai", "pydantic", "httpx")
            },
        }
        self.flush_manifest()

    def flush_manifest(self) -> None:
        write_json(self.directory / "run.json", self._manifest)

    def update_manifest(self, **values: Any) -> None:
        self._manifest.update(values)
        if values.get("status") not in (None, "running"):
            self._manifest.setdefault("completed_at", utc_now())
            self._manifest["elapsed_seconds"] = round(
                time.monotonic() - self._started_monotonic, 3
            )
        self.flush_manifest()

    @staticmethod
    def _package_version(package: str) -> str | None:
        try:
            return version(package)
        except PackageNotFoundError:
            return None

    def write_baseline(self, comparison: str) -> None:
        atomic_write(self.directory / "baseline.compare.txt", comparison)

    def write_selected(
        self,
        candidate: Candidate,
        score: float,
        changed_files: list[str],
    ) -> None:
        write_json(
            self.directory / "selected-change-set.json",
            {
                "candidate": candidate.model_dump(mode="json"),
                "changed_files": changed_files,
                "score": score,
            },
        )

    def write_prompt(self, iteration: int, prompt: str) -> None:
        atomic_write(self.directory / ("iteration-%02d.prompt.txt" % iteration), prompt)

    def write_generation(
        self,
        iteration: int,
        batch: CandidateBatch,
        completion: Any,
    ) -> None:
        write_json(self.directory / ("iteration-%02d.batch.json" % iteration), batch)
        if hasattr(completion, "model_dump"):
            completion = completion.model_dump(mode="json")
        write_json(
            self.directory / ("iteration-%02d.completion.json" % iteration),
            completion,
        )

    def write_candidate(
        self,
        iteration: int,
        index: int,
        candidate: Candidate,
        comparison: str,
    ) -> None:
        prefix = self.directory / ("iteration-%02d-candidate-%02d" % (iteration, index))
        write_json(prefix.with_suffix(".json"), candidate)
        atomic_write(prefix.with_suffix(".c"), candidate.source.rstrip() + "\n")
        atomic_write(prefix.with_suffix(".compare.txt"), comparison.rstrip() + "\n")

    def write_repair_prompt(
        self,
        iteration: int,
        index: int,
        repair_attempt: int,
        prompt: str,
    ) -> None:
        prefix = self._repair_prefix(iteration, index, repair_attempt)
        atomic_write(prefix.with_suffix(".prompt.txt"), prompt)

    def write_repair_generation(
        self,
        iteration: int,
        index: int,
        repair_attempt: int,
        candidate: Candidate,
        completion: Any,
    ) -> None:
        prefix = self._repair_prefix(iteration, index, repair_attempt)
        if hasattr(completion, "model_dump"):
            completion = completion.model_dump(mode="json")
        write_json(prefix.with_suffix(".candidate.json"), candidate)
        write_json(prefix.with_suffix(".completion.json"), completion)

    def write_symbol_repair_generation(
        self,
        iteration: int,
        index: int,
        repair_attempt: int,
        proposals: SymbolProposalBatch,
        selected: str | None,
        candidate: Candidate | None,
        completion: Any,
    ) -> None:
        prefix = self._repair_prefix(iteration, index, repair_attempt)
        if hasattr(completion, "model_dump"):
            completion = completion.model_dump(mode="json")
        write_json(
            prefix.with_suffix(".symbols.json"),
            {"proposals": proposals.symbols, "selected": selected},
        )
        if candidate is not None:
            write_json(prefix.with_suffix(".candidate.json"), candidate)
        write_json(prefix.with_suffix(".completion.json"), completion)

    def write_repair_candidate(
        self,
        iteration: int,
        index: int,
        repair_attempt: int,
        candidate: Candidate,
        comparison: str,
    ) -> None:
        prefix = self._repair_prefix(iteration, index, repair_attempt)
        write_json(prefix.with_suffix(".json"), candidate)
        atomic_write(prefix.with_suffix(".c"), candidate.source.rstrip() + "\n")
        atomic_write(prefix.with_suffix(".compare.txt"), comparison.rstrip() + "\n")

    def _repair_prefix(
        self,
        iteration: int,
        index: int,
        repair_attempt: int,
    ) -> Path:
        return self.directory / (
            "iteration-%02d-candidate-%02d-repair-%02d"
            % (iteration, index, repair_attempt)
        )

"""File-backed audit log for every deterministic and model-assisted stage."""

from __future__ import annotations

import re
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .models import (
    Candidate,
    ContractProposal,
    SearchConfig,
    SourcePatch,
    SymbolProposalBatch,
    TargetSpec,
)
from .utils import atomic_write, utc_now, write_json


class RunLog:
    def __init__(self, output_root: Path, target: TargetSpec, config: SearchConfig):
        subsecond = (time.time_ns() % 1_000_000_000) // 1_000
        stamp = "%s-%06d" % (time.strftime("%Y%m%d-%H%M%S"), subsecond)
        model_name = re.sub(r"[^A-Za-z0-9._-]+", "-", config.model).strip("-.")
        model_name = model_name[:80] or "local-model"
        self.directory = output_root / model_name / ("%08X" % target.address) / stamp
        self.directory.mkdir(parents=True, exist_ok=False)
        self._started_monotonic = time.monotonic()
        self._manifest: dict[str, Any] = {
            "started_at": utc_now(),
            "status": "running",
            "workflow": "ghidra-seed-bounded-qwen-edits-v2",
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

    @staticmethod
    def _completion(completion: Any) -> Any:
        return (
            completion.model_dump(mode="json")
            if hasattr(completion, "model_dump")
            else completion
        )

    def write_baseline(self, comparison: str) -> None:
        atomic_write(self.directory / "baseline.compare.txt", comparison)

    def write_contract_prompt(self, prompt: str) -> None:
        atomic_write(self.directory / "contract.prompt.txt", prompt)

    def write_contract_generation(
        self,
        contract: ContractProposal,
        completion: Any,
    ) -> None:
        write_json(
            self.directory / "contract.json",
            {
                "contract": contract.model_dump(mode="json"),
                "completion": self._completion(completion),
            },
        )

    def write_symbol_prompt(self, attempt: int, prompt: str) -> None:
        atomic_write(
            self.directory / ("contract-name-repair-%02d.prompt.txt" % attempt),
            prompt,
        )

    def write_symbol_generation(
        self,
        attempt: int,
        proposals: SymbolProposalBatch,
        selected: str | None,
        completion: Any,
    ) -> None:
        write_json(
            self.directory / ("contract-name-repair-%02d.json" % attempt),
            {
                "proposals": proposals.symbols,
                "selected": selected,
                "completion": self._completion(completion),
            },
        )

    def write_seed(
        self,
        candidate: Candidate,
        mechanical_changes: list[str],
        comparison: str,
        score: float | None,
        origin: str,
    ) -> None:
        write_json(
            self.directory / "seed.json",
            {
                "candidate": candidate.model_dump(mode="json"),
                "mechanical_changes": mechanical_changes,
                "origin": origin,
                "score": score,
            },
        )
        atomic_write(self.directory / "seed.c", candidate.source.rstrip() + "\n")
        atomic_write(self.directory / "seed.compare.txt", comparison.rstrip() + "\n")

    def write_edit_prompt(self, round_number: int, kind: str, prompt: str) -> None:
        atomic_write(
            self.directory / ("edit-%02d.%s.prompt.txt" % (round_number, kind)),
            prompt,
        )

    def write_edit_generation(
        self,
        round_number: int,
        kind: str,
        patch: SourcePatch,
        completion: Any,
    ) -> None:
        write_json(
            self.directory / ("edit-%02d.%s.response.json" % (round_number, kind)),
            {
                "patch": patch.model_dump(mode="json"),
                "completion": self._completion(completion),
            },
        )

    def write_edit_result(
        self,
        round_number: int,
        kind: str,
        candidate: Candidate | None,
        patch: SourcePatch,
        rejected_operations: list[str],
        comparison: str,
        score: float | None,
        followed: bool,
        metrics: dict[str, int],
    ) -> None:
        prefix = self.directory / ("edit-%02d.%s" % (round_number, kind))
        write_json(
            Path(str(prefix) + ".result.json"),
            {
                "followed": followed,
                "candidate": (
                    candidate.model_dump(mode="json") if candidate is not None else None
                ),
                "metrics": metrics,
                "patch": patch.model_dump(mode="json"),
                "rejected_operations": rejected_operations,
                "score": score,
            },
        )
        if candidate is not None:
            atomic_write(
                Path(str(prefix) + ".candidate.c"),
                candidate.source.rstrip() + "\n",
            )
            write_json(Path(str(prefix) + ".candidate.json"), candidate)
        atomic_write(Path(str(prefix) + ".compare.txt"), comparison.rstrip() + "\n")

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

"""Bounded generate/compile/compare search controlled entirely by Python."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .llama_server import ManagedLlamaServer
from .model_client import CandidateGenerator
from .models import Candidate, LlamaServerConfig, SearchConfig, TargetSpec
from .prompts import (
    HistoryItem,
    build_compile_repair_prompt,
    build_prompt,
    build_symbol_repair_prompt,
    build_validation_repair_prompt,
)
from .repository import (
    ProjectRepository,
    candidate_fingerprint,
    current_function,
    is_generic_function_symbol,
    normalize_candidate_marker,
    rename_candidate_symbol,
    validate_candidate,
)
from .runlog import RunLog


@dataclass(frozen=True)
class SearchResult:
    score: float | None
    target_reached: bool
    attempts: int
    session_directory: Path
    symbol: str | None = None
    prototype: str | None = None


@dataclass(frozen=True)
class _AttemptResult:
    score: float | None
    feedback: str


def _select_symbol_proposal(
    proposals: list[str],
    reserved_symbols: set[str],
    address: int,
) -> str | None:
    address_text = "%X" % address
    for symbol in proposals:
        if symbol in reserved_symbols:
            continue
        if is_generic_function_symbol(symbol):
            continue
        if address_text in symbol.upper():
            continue
        return symbol
    return None


class ReconstructionSearch:
    def __init__(
        self,
        repository: ProjectRepository,
        target: TargetSpec,
        search_config: SearchConfig,
        server_config: LlamaServerConfig,
    ):
        self.repository = repository
        self.target = target
        self.config = search_config
        self.server_config = server_config

    def _compile_candidate(
        self,
        candidate: Candidate,
        candidate_target: TargetSpec,
        baseline_workspace: dict[Path, str],
        best_workspace: dict[Path, str],
    ) -> tuple[dict[Path, str], float | None, str]:
        trial_workspace = self.repository.render_candidate_workspace(
            candidate_target,
            candidate,
            baseline_workspace,
        )
        try:
            self.repository.apply_workspace(trial_workspace)
            score, comparison = self.repository.compare(
                candidate_target, self.config.build_timeout
            )
        finally:
            self.repository.apply_workspace(best_workspace)
        return trial_workspace, score, comparison

    def run(self, dry_run_prompt: bool = False) -> SearchResult:
        evidence = self.repository.collect_evidence(
            self.target, self.config.max_callees
        )
        baseline_workspace = self.repository.snapshot_workspace(self.target)
        output_root = self.repository.config.resolve(
            self.repository.root, self.repository.config.output_dir
        )
        run_log = RunLog(output_root, self.target, self.config)
        best_workspace = dict(baseline_workspace)
        existing_source = current_function(
            baseline_workspace[self.target.source_path.resolve()], self.target.address
        )
        best_candidate = (
            Candidate(
                symbol=self.target.symbol,
                prototype=self.target.prototype,
                source=existing_source,
            )
            if existing_source is not None and self.target.has_contract
            else None
        )
        best_target = self.target
        best_score = -1.0
        best_feedback = ""
        history: list[HistoryItem] = []
        seen: set[str] = set()
        reserved_symbols = set(self.repository.reserved_symbols(self.target))
        allowed_support_paths = self.repository.allowed_support_paths()
        attempts = 0

        try:
            if best_candidate is not None:
                baseline_score, baseline_output = self.repository.compare(
                    self.target, self.config.build_timeout
                )
                run_log.write_baseline(baseline_output)
                fingerprint = candidate_fingerprint(best_candidate)
                seen.add(fingerprint)
                history.append(
                    HistoryItem(
                        fingerprint=fingerprint,
                        score=baseline_score,
                        candidate=best_candidate.model_dump_json(indent=2),
                    )
                )
                best_feedback = self.repository.compact_feedback(
                    baseline_output, baseline_score
                )
                if baseline_score is not None:
                    best_score = baseline_score
                    print("baseline similarity: %.2f%%" % baseline_score, flush=True)
                    if baseline_score >= self.config.target_score:
                        result = SearchResult(
                            score=baseline_score,
                            target_reached=True,
                            attempts=0,
                            session_directory=run_log.directory,
                            symbol=best_target.symbol,
                            prototype=best_target.prototype,
                        )
                        run_log.update_manifest(
                            status="complete",
                            completed_without_model=True,
                            result={
                                "score": result.score,
                                "target_reached": True,
                                "attempts": 0,
                                "symbol": best_target.symbol,
                                "prototype": best_target.prototype,
                            },
                        )
                        print("target already meets %.2f%%" % self.config.target_score)
                        print("logs: %s" % run_log.directory)
                        return result

            initial_prompt = build_prompt(
                self.target,
                evidence,
                self.config,
                best_candidate,
                best_feedback,
                history,
            )
            if dry_run_prompt:
                run_log.write_prompt(1, initial_prompt)
                run_log.update_manifest(status="dry-run", stage="prompt")
                print(run_log.directory / "iteration-01.prompt.txt")
                return SearchResult(
                    score=None if best_score < 0 else best_score,
                    target_reached=False,
                    attempts=0,
                    session_directory=run_log.directory,
                    symbol=best_target.symbol,
                    prototype=best_target.prototype,
                )

            server = ManagedLlamaServer(
                self.server_config, self.config, run_log.directory
            )
            with server:
                generator = CandidateGenerator(server, self.config)
                try:
                    for iteration in range(1, self.config.max_iterations + 1):
                        prompt = build_prompt(
                            best_target,
                            evidence,
                            self.config,
                            best_candidate,
                            best_feedback,
                            history,
                        )
                        run_log.write_prompt(iteration, prompt)
                        print(
                            "iteration %d/%d: requesting %s"
                            % (
                                iteration,
                                self.config.max_iterations,
                                self.config.model,
                            ),
                            flush=True,
                        )
                        batch, completion = generator.generate(prompt, iteration)
                        run_log.write_generation(iteration, batch, completion)

                        batch_results: list[_AttemptResult] = []
                        target_reached = False
                        batch_contract = best_target
                        candidates = batch.candidates[
                            : self.config.candidates_per_iteration
                        ]
                        for index, proposed in enumerate(candidates, 1):
                            candidate = normalize_candidate_marker(
                                proposed, self.target.address
                            )
                            contract_locked = batch_contract.has_contract
                            candidate_target = (
                                batch_contract
                                if contract_locked
                                else self.target.with_candidate_contract(candidate)
                            )
                            fingerprint = candidate_fingerprint(candidate)
                            if fingerprint in seen:
                                feedback = (
                                    "CANDIDATE SKIPPED: fingerprint already tried"
                                )
                                run_log.write_candidate(
                                    iteration, index, candidate, feedback
                                )
                                batch_results.append(_AttemptResult(None, feedback))
                                print(
                                    "iteration %d candidate %d: duplicate"
                                    % (iteration, index),
                                    flush=True,
                                )
                                continue
                            seen.add(fingerprint)
                            attempts += 1

                            repair_attempts_used = 0
                            candidate_valid = False
                            while True:
                                try:
                                    validate_candidate(
                                        candidate,
                                        candidate_target,
                                        reserved_symbols,
                                        allowed_support_paths,
                                    )
                                except ValueError as error:
                                    feedback = (
                                        "CANDIDATE REJECTED BEFORE BUILD: %s" % error
                                    )
                                    if repair_attempts_used == 0:
                                        run_log.write_candidate(
                                            iteration, index, candidate, feedback
                                        )
                                    else:
                                        run_log.write_repair_candidate(
                                            iteration,
                                            index,
                                            repair_attempts_used,
                                            candidate,
                                            feedback,
                                        )
                                    history.append(
                                        HistoryItem(
                                            fingerprint=fingerprint,
                                            score=None,
                                            candidate=candidate.model_dump_json(
                                                indent=2
                                            ),
                                        )
                                    )
                                    if (
                                        repair_attempts_used
                                        >= self.config.compile_repair_attempts
                                    ):
                                        break

                                    repair_attempt = repair_attempts_used + 1
                                    symbol_repair = not contract_locked and (
                                        candidate.symbol in reserved_symbols
                                        or is_generic_function_symbol(candidate.symbol)
                                    )
                                    if symbol_repair:
                                        repair_prompt = build_symbol_repair_prompt(
                                            batch_contract,
                                            evidence,
                                            candidate,
                                            feedback,
                                        )
                                        repair_kind = "symbol"
                                    else:
                                        repair_prompt = build_validation_repair_prompt(
                                            batch_contract,
                                            evidence,
                                            self.config,
                                            candidate,
                                            feedback,
                                            repair_attempt,
                                        )
                                        repair_kind = "validation"
                                    run_log.write_repair_prompt(
                                        iteration,
                                        index,
                                        repair_attempt,
                                        repair_prompt,
                                    )
                                    print(
                                        "iteration %d candidate %d: requesting "
                                        "%s repair %d/%d"
                                        % (
                                            iteration,
                                            index,
                                            repair_kind,
                                            repair_attempt,
                                            self.config.compile_repair_attempts,
                                        ),
                                        flush=True,
                                    )
                                    if symbol_repair:
                                        proposals, repair_completion = (
                                            generator.propose_symbols(
                                                repair_prompt,
                                                iteration,
                                                index,
                                                repair_attempt,
                                            )
                                        )
                                        selected_symbol = _select_symbol_proposal(
                                            proposals.symbols,
                                            reserved_symbols,
                                            self.target.address,
                                        )
                                        repaired_candidate = (
                                            rename_candidate_symbol(
                                                candidate, selected_symbol
                                            )
                                            if selected_symbol is not None
                                            else None
                                        )
                                        run_log.write_symbol_repair_generation(
                                            iteration,
                                            index,
                                            repair_attempt,
                                            proposals,
                                            selected_symbol,
                                            repaired_candidate,
                                            repair_completion,
                                        )
                                        repair_attempts_used = repair_attempt
                                        if repaired_candidate is None:
                                            feedback = (
                                                "SYMBOL REPAIR REJECTED: every "
                                                "proposal is reserved or invalid"
                                            )
                                            print(
                                                "iteration %d candidate %d repair "
                                                "%d: no usable symbol"
                                                % (
                                                    iteration,
                                                    index,
                                                    repair_attempt,
                                                ),
                                                flush=True,
                                            )
                                            continue
                                    else:
                                        repaired, repair_completion = generator.repair(
                                            repair_prompt,
                                            iteration,
                                            index,
                                            repair_attempt,
                                        )
                                        run_log.write_repair_generation(
                                            iteration,
                                            index,
                                            repair_attempt,
                                            repaired,
                                            repair_completion,
                                        )
                                        repaired_candidate = normalize_candidate_marker(
                                            repaired, self.target.address
                                        )
                                    repaired_fingerprint = candidate_fingerprint(
                                        repaired_candidate
                                    )
                                    repair_attempts_used = repair_attempt
                                    if repaired_fingerprint in seen:
                                        duplicate_feedback = (
                                            "REPAIR SKIPPED: fingerprint already tried"
                                        )
                                        run_log.write_repair_candidate(
                                            iteration,
                                            index,
                                            repair_attempt,
                                            repaired_candidate,
                                            duplicate_feedback,
                                        )
                                        print(
                                            "iteration %d candidate %d repair %d: "
                                            "duplicate"
                                            % (iteration, index, repair_attempt),
                                            flush=True,
                                        )
                                        continue
                                    seen.add(repaired_fingerprint)
                                    attempts += 1
                                    candidate = repaired_candidate
                                    fingerprint = repaired_fingerprint
                                    if not contract_locked:
                                        candidate_target = (
                                            self.target.with_candidate_contract(
                                                candidate
                                            )
                                        )
                                    continue
                                candidate_valid = True
                                break

                            if not candidate_valid:
                                batch_results.append(_AttemptResult(None, feedback))
                                print(
                                    "iteration %d candidate %d: rejected"
                                    % (iteration, index),
                                    flush=True,
                                )
                                continue

                            trial_workspace, score, comparison = (
                                self._compile_candidate(
                                    candidate,
                                    candidate_target,
                                    baseline_workspace,
                                    best_workspace,
                                )
                            )
                            feedback = self.repository.compact_feedback(
                                comparison, score
                            )
                            if repair_attempts_used == 0:
                                run_log.write_candidate(
                                    iteration, index, candidate, comparison
                                )
                            else:
                                run_log.write_repair_candidate(
                                    iteration,
                                    index,
                                    repair_attempts_used,
                                    candidate,
                                    comparison,
                                )
                            history.append(
                                HistoryItem(
                                    fingerprint=fingerprint,
                                    score=score,
                                    candidate=candidate.model_dump_json(indent=2),
                                )
                            )

                            if (
                                score is None
                                and self.config.compile_repair_attempts
                                > repair_attempts_used
                                and self.repository.is_repairable_build_failure(
                                    comparison
                                )
                            ):
                                for repair_attempt in range(
                                    repair_attempts_used + 1,
                                    self.config.compile_repair_attempts + 1,
                                ):
                                    repair_prompt = build_compile_repair_prompt(
                                        candidate_target,
                                        evidence,
                                        self.config,
                                        candidate,
                                        feedback,
                                        repair_attempt,
                                    )
                                    run_log.write_repair_prompt(
                                        iteration,
                                        index,
                                        repair_attempt,
                                        repair_prompt,
                                    )
                                    print(
                                        "iteration %d candidate %d: requesting "
                                        "compile repair %d/%d"
                                        % (
                                            iteration,
                                            index,
                                            repair_attempt,
                                            self.config.compile_repair_attempts,
                                        ),
                                        flush=True,
                                    )
                                    repaired, repair_completion = generator.repair(
                                        repair_prompt,
                                        iteration,
                                        index,
                                        repair_attempt,
                                    )
                                    run_log.write_repair_generation(
                                        iteration,
                                        index,
                                        repair_attempt,
                                        repaired,
                                        repair_completion,
                                    )

                                    repaired_candidate = normalize_candidate_marker(
                                        repaired, self.target.address
                                    )
                                    repaired_fingerprint = candidate_fingerprint(
                                        repaired_candidate
                                    )
                                    if repaired_fingerprint in seen:
                                        duplicate_feedback = (
                                            "REPAIR SKIPPED: fingerprint already tried"
                                        )
                                        run_log.write_repair_candidate(
                                            iteration,
                                            index,
                                            repair_attempt,
                                            repaired_candidate,
                                            duplicate_feedback,
                                        )
                                        print(
                                            "iteration %d candidate %d repair %d: "
                                            "duplicate"
                                            % (iteration, index, repair_attempt),
                                            flush=True,
                                        )
                                        continue

                                    seen.add(repaired_fingerprint)
                                    attempts += 1
                                    candidate = repaired_candidate
                                    fingerprint = repaired_fingerprint
                                    try:
                                        validate_candidate(
                                            candidate,
                                            candidate_target,
                                            reserved_symbols,
                                            allowed_support_paths,
                                        )
                                    except ValueError as error:
                                        score = None
                                        feedback = (
                                            "REPAIR REJECTED BEFORE BUILD: %s" % error
                                        )
                                        run_log.write_repair_candidate(
                                            iteration,
                                            index,
                                            repair_attempt,
                                            candidate,
                                            feedback,
                                        )
                                        history.append(
                                            HistoryItem(
                                                fingerprint=fingerprint,
                                                score=None,
                                                candidate=candidate.model_dump_json(
                                                    indent=2
                                                ),
                                            )
                                        )
                                        print(
                                            "iteration %d candidate %d repair %d: "
                                            "rejected"
                                            % (iteration, index, repair_attempt),
                                            flush=True,
                                        )
                                        continue

                                    trial_workspace, score, comparison = (
                                        self._compile_candidate(
                                            candidate,
                                            candidate_target,
                                            baseline_workspace,
                                            best_workspace,
                                        )
                                    )
                                    feedback = self.repository.compact_feedback(
                                        comparison, score
                                    )
                                    run_log.write_repair_candidate(
                                        iteration,
                                        index,
                                        repair_attempt,
                                        candidate,
                                        comparison,
                                    )
                                    history.append(
                                        HistoryItem(
                                            fingerprint=fingerprint,
                                            score=score,
                                            candidate=candidate.model_dump_json(
                                                indent=2
                                            ),
                                        )
                                    )
                                    if score is not None:
                                        print(
                                            "iteration %d candidate %d repair %d: "
                                            "similarity %.2f%%"
                                            % (
                                                iteration,
                                                index,
                                                repair_attempt,
                                                score,
                                            ),
                                            flush=True,
                                        )
                                        break
                                    if not self.repository.is_repairable_build_failure(
                                        comparison
                                    ):
                                        print(
                                            "iteration %d candidate %d repair %d: "
                                            "build/compare failed"
                                            % (iteration, index, repair_attempt),
                                            flush=True,
                                        )
                                        break

                            batch_results.append(_AttemptResult(score, feedback))

                            if score is None:
                                print(
                                    "iteration %d candidate %d: build/compare failed"
                                    % (iteration, index),
                                    flush=True,
                                )
                                continue
                            print(
                                "iteration %d candidate %d: similarity %.2f%%"
                                % (iteration, index, score),
                                flush=True,
                            )
                            if score > best_score:
                                best_score = score
                                best_workspace = trial_workspace
                                best_candidate = candidate
                                best_target = candidate_target
                                best_feedback = feedback
                                self.repository.apply_workspace(best_workspace)
                            if best_score >= self.config.target_score:
                                target_reached = True
                                break

                        if batch_results:
                            scored = [
                                result
                                for result in batch_results
                                if result.score is not None
                            ]
                            selected = (
                                max(scored, key=lambda result: result.score)
                                if scored
                                else batch_results[-1]
                            )
                            summary = "\n".join(
                                "candidate %d: %s"
                                % (
                                    index,
                                    "failed/rejected"
                                    if result.score is None
                                    else "%.2f%%" % result.score,
                                )
                                for index, result in enumerate(batch_results, 1)
                            )
                            best_feedback = (
                                selected.feedback + "\nBATCH SUMMARY\n" + summary
                            )

                        if target_reached:
                            assert best_candidate is not None
                            changed_files = self.repository.changed_workspace_files(
                                baseline_workspace, best_workspace
                            )
                            run_log.write_selected(
                                best_candidate, best_score, changed_files
                            )
                            result = SearchResult(
                                score=best_score,
                                target_reached=True,
                                attempts=attempts,
                                session_directory=run_log.directory,
                                symbol=best_target.symbol,
                                prototype=best_target.prototype,
                            )
                            run_log.update_manifest(
                                status="complete",
                                result={
                                    "score": best_score,
                                    "target_reached": True,
                                    "attempts": attempts,
                                    "symbol": best_target.symbol,
                                    "prototype": best_target.prototype,
                                    "changed_files": changed_files,
                                },
                            )
                            print("target reached: %.2f%%" % best_score, flush=True)
                            print(
                                "selected contract: %s — %s"
                                % (best_target.symbol, best_target.prototype),
                                flush=True,
                            )
                            print("logs: %s" % run_log.directory)
                            return result
                finally:
                    generator.close()

            score = None if best_score < 0 else best_score
            result = SearchResult(
                score=score,
                target_reached=False,
                attempts=attempts,
                session_directory=run_log.directory,
                symbol=best_target.symbol,
                prototype=best_target.prototype,
            )
            changed_files: list[str] = []
            if score is not None and best_candidate is not None:
                changed_files = self.repository.changed_workspace_files(
                    baseline_workspace, best_workspace
                )
                run_log.write_selected(best_candidate, score, changed_files)
            run_log.update_manifest(
                status="complete" if score is not None else "failed",
                result={
                    "score": score,
                    "target_reached": False,
                    "attempts": attempts,
                    "symbol": best_target.symbol,
                    "prototype": best_target.prototype,
                    "changed_files": changed_files,
                },
            )
            if score is None:
                print(
                    "no candidate compiled; original source restored",
                    flush=True,
                )
            else:
                print("best retained similarity: %.2f%%" % score)
                print(
                    "selected contract: %s — %s"
                    % (best_target.symbol, best_target.prototype),
                    flush=True,
                )
            print("logs: %s" % run_log.directory)
            return result
        except BaseException as error:
            run_log.update_manifest(
                status="failed",
                error={"type": type(error).__name__, "message": str(error)},
            )
            raise
        finally:
            self.repository.apply_workspace(best_workspace)

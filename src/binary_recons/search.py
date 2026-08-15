"""Bounded generate/compile/compare search controlled entirely by Python."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .llama_server import ManagedLlamaServer
from .model_client import CandidateGenerator
from .models import Candidate, LlamaServerConfig, SearchConfig, TargetSpec
from .prompts import HistoryItem, build_compile_repair_prompt, build_prompt
from .repository import (
    ProjectRepository,
    candidate_fingerprint,
    current_function,
    replace_or_insert_function,
    validate_candidate,
)
from .runlog import RunLog
from .utils import atomic_write, read_text


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
        best_source: str,
    ) -> tuple[str, float | None, str]:
        trial_source = replace_or_insert_function(
            best_source, self.target.address, candidate.source
        )
        atomic_write(self.target.source_path, trial_source)
        try:
            score, comparison = self.repository.compare(
                candidate_target, self.config.build_timeout
            )
        finally:
            atomic_write(self.target.source_path, best_source)
        return trial_source, score, comparison

    def run(self, dry_run_prompt: bool = False) -> SearchResult:
        evidence = self.repository.collect_evidence(
            self.target, self.config.max_callees
        )
        output_root = self.repository.config.resolve(
            self.repository.root, self.repository.config.output_dir
        )
        run_log = RunLog(output_root, self.target, self.config)
        original_source = read_text(self.target.source_path)
        best_source = original_source
        best_candidate = current_function(original_source, self.target.address)
        best_target = self.target
        best_score = -1.0
        best_feedback = ""
        history: list[HistoryItem] = []
        seen: set[str] = set()
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
                        candidate=best_candidate,
                    )
                )
                if baseline_score is not None:
                    best_score = baseline_score
                    best_feedback = self.repository.compact_feedback(
                        baseline_output, baseline_score
                    )
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
                            candidate = proposed
                            candidate_target = (
                                batch_contract
                                if batch_contract.has_contract
                                else self.target.with_candidate_contract(candidate)
                            )
                            fingerprint = candidate_fingerprint(candidate.source)
                            if fingerprint in seen:
                                feedback = (
                                    "CANDIDATE SKIPPED: fingerprint already tried"
                                )
                                run_log.write_candidate(
                                    iteration, index, candidate.source, feedback
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

                            try:
                                validate_candidate(candidate, candidate_target)
                            except ValueError as error:
                                feedback = "CANDIDATE REJECTED BEFORE BUILD: %s" % error
                                run_log.write_candidate(
                                    iteration, index, candidate.source, feedback
                                )
                                history.append(
                                    HistoryItem(
                                        fingerprint=fingerprint,
                                        score=None,
                                        candidate=candidate.source,
                                    )
                                )
                                batch_results.append(_AttemptResult(None, feedback))
                                print(
                                    "iteration %d candidate %d: rejected"
                                    % (iteration, index),
                                    flush=True,
                                )
                                continue

                            trial_source, score, comparison = self._compile_candidate(
                                candidate, candidate_target, best_source
                            )
                            feedback = self.repository.compact_feedback(
                                comparison, score
                            )
                            run_log.write_candidate(
                                iteration, index, candidate.source, comparison
                            )
                            history.append(
                                HistoryItem(
                                    fingerprint=fingerprint,
                                    score=score,
                                    candidate=candidate.source,
                                )
                            )

                            if (
                                score is None
                                and self.config.compile_repair_attempts > 0
                                and self.repository.has_compiler_errors(comparison)
                            ):
                                for repair_attempt in range(
                                    1, self.config.compile_repair_attempts + 1
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

                                    repaired_candidate = repaired
                                    repaired_fingerprint = candidate_fingerprint(
                                        repaired_candidate.source
                                    )
                                    if repaired_fingerprint in seen:
                                        feedback = (
                                            "REPAIR SKIPPED: fingerprint already tried"
                                        )
                                        run_log.write_repair_candidate(
                                            iteration,
                                            index,
                                            repair_attempt,
                                            repaired_candidate.source,
                                            feedback,
                                        )
                                        print(
                                            "iteration %d candidate %d repair %d: "
                                            "duplicate"
                                            % (iteration, index, repair_attempt),
                                            flush=True,
                                        )
                                        break

                                    seen.add(repaired_fingerprint)
                                    attempts += 1
                                    candidate = repaired_candidate
                                    fingerprint = repaired_fingerprint
                                    try:
                                        validate_candidate(candidate, candidate_target)
                                    except ValueError as error:
                                        score = None
                                        feedback = (
                                            "REPAIR REJECTED BEFORE BUILD: %s" % error
                                        )
                                        run_log.write_repair_candidate(
                                            iteration,
                                            index,
                                            repair_attempt,
                                            candidate.source,
                                            feedback,
                                        )
                                        history.append(
                                            HistoryItem(
                                                fingerprint=fingerprint,
                                                score=None,
                                                candidate=candidate.source,
                                            )
                                        )
                                        print(
                                            "iteration %d candidate %d repair %d: "
                                            "rejected"
                                            % (iteration, index, repair_attempt),
                                            flush=True,
                                        )
                                        continue

                                    trial_source, score, comparison = (
                                        self._compile_candidate(
                                            candidate, candidate_target, best_source
                                        )
                                    )
                                    feedback = self.repository.compact_feedback(
                                        comparison, score
                                    )
                                    run_log.write_repair_candidate(
                                        iteration,
                                        index,
                                        repair_attempt,
                                        candidate.source,
                                        comparison,
                                    )
                                    history.append(
                                        HistoryItem(
                                            fingerprint=fingerprint,
                                            score=score,
                                            candidate=candidate.source,
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
                                    if not self.repository.has_compiler_errors(
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
                                best_source = trial_source
                                best_candidate = candidate.source
                                best_target = candidate_target
                                best_feedback = feedback
                                atomic_write(self.target.source_path, best_source)
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
                            result = SearchResult(
                                score=best_score,
                                target_reached=True,
                                attempts=attempts,
                                session_directory=run_log.directory,
                                symbol=best_target.symbol,
                                prototype=best_target.prototype,
                            )
                            self.repository.persist_contract(best_target)
                            run_log.update_manifest(
                                status="complete",
                                result={
                                    "score": best_score,
                                    "target_reached": True,
                                    "attempts": attempts,
                                    "symbol": best_target.symbol,
                                    "prototype": best_target.prototype,
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
            if score is not None:
                self.repository.persist_contract(best_target)
            run_log.update_manifest(
                status="complete" if score is not None else "failed",
                result={
                    "score": score,
                    "target_reached": False,
                    "attempts": attempts,
                    "symbol": best_target.symbol,
                    "prototype": best_target.prototype,
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
            atomic_write(self.target.source_path, best_source)

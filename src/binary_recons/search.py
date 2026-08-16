"""Compile-first Ghidra integration with bounded model-edit trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .llama_server import ManagedLlamaServer
from .model_client import ModelRequestError, StructuredModelClient
from .models import (
    Candidate,
    ContractProposal,
    EvidenceBundle,
    LlamaServerConfig,
    SearchConfig,
    SimilarityPatch,
    SourcePatch,
    TargetSpec,
)
from .prompts import (
    build_compile_patch_prompt,
    build_contract_prompt,
    build_similarity_patch_prompt,
    build_symbol_repair_prompt,
)
from .repository import (
    ProjectRepository,
    current_function,
    is_generic_function_symbol,
    rename_candidate_symbol,
    source_safety_feedback,
    validate_candidate,
)
from .runlog import RunLog
from .seed import (
    candidate_from_seed,
    normalize_contract,
    normalize_decompiler_seed,
    normalize_resumed_candidate,
)
from .source_edits import (
    apply_source_patch,
    bind_supporting_address_symbols,
    patch_metrics,
    sanitize_source_patch,
)


@dataclass(frozen=True)
class SearchResult:
    score: float | None
    target_reached: bool
    attempts: int
    session_directory: Path
    symbol: str | None = None
    prototype: str | None = None


@dataclass(frozen=True)
class _Evaluation:
    candidate: Candidate
    target: TargetSpec
    workspace: dict[Path, str]
    score: float | None
    output: str


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


def _contract_problem(
    contract: ContractProposal,
    reserved_symbols: set[str],
    address: int,
) -> str | None:
    if contract.symbol in reserved_symbols:
        return "The proposed symbol `%s` is already reserved." % contract.symbol
    if is_generic_function_symbol(contract.symbol):
        return "The proposed symbol `%s` is operational or generic." % contract.symbol
    if "%X" % address in contract.symbol.upper():
        return "The proposed symbol `%s` contains the target address." % contract.symbol
    return None


def _rename_contract(contract: ContractProposal, symbol: str) -> ContractProposal:
    prototype, count = re.subn(
        r"\b%s\b" % re.escape(contract.symbol),
        symbol,
        contract.prototype,
    )
    if count != 1:
        raise RuntimeError("could not rename the inferred function contract safely")
    return ContractProposal(symbol=symbol, prototype=prototype)


class ReconstructionSearch:
    def __init__(
        self,
        repository: ProjectRepository,
        target: TargetSpec,
        search_config: SearchConfig,
        server_config: LlamaServerConfig,
        initial_candidate: Candidate | None = None,
    ):
        self.repository = repository
        self.target = target
        self.config = search_config
        self.server_config = server_config
        self.initial_candidate = initial_candidate

    def _evaluate(
        self,
        candidate: Candidate,
        active_target: TargetSpec,
        baseline_workspace: dict[Path, str],
        restore_workspace: dict[Path, str],
        reserved_symbols: set[str],
        allowed_support_paths: set[str],
    ) -> _Evaluation:
        validate_candidate(
            candidate,
            active_target,
            reserved_symbols,
            allowed_support_paths,
        )
        trial_workspace = self.repository.render_candidate_workspace(
            active_target,
            candidate,
            baseline_workspace,
        )
        safety_feedback = source_safety_feedback(candidate)
        if safety_feedback is not None:
            return _Evaluation(
                candidate,
                active_target,
                trial_workspace,
                None,
                safety_feedback,
            )
        try:
            self.repository.apply_workspace(trial_workspace)
            score, output = self.repository.compare(
                active_target,
                self.config.build_timeout,
            )
        finally:
            self.repository.apply_workspace(restore_workspace)
        return _Evaluation(candidate, active_target, trial_workspace, score, output)

    def _repair_contract_name(
        self,
        client: StructuredModelClient,
        run_log: RunLog,
        evidence: EvidenceBundle,
        contract: ContractProposal,
        reserved_symbols: set[str],
    ) -> ContractProposal:
        problem = _contract_problem(contract, reserved_symbols, self.target.address)
        for attempt in range(1, 3):
            if problem is None:
                return contract
            prompt = build_symbol_repair_prompt(
                self.target,
                evidence,
                contract,
                problem,
            )
            run_log.write_symbol_prompt(attempt, prompt)
            proposals, completion = client.propose_symbols(prompt, attempt)
            selected = _select_symbol_proposal(
                proposals.symbols,
                reserved_symbols,
                self.target.address,
            )
            run_log.write_symbol_generation(
                attempt,
                proposals,
                selected,
                completion,
            )
            if selected is not None:
                contract = _rename_contract(contract, selected)
            problem = _contract_problem(contract, reserved_symbols, self.target.address)
        if problem is not None:
            raise RuntimeError(
                "Qwen could not produce a usable function name: %s" % problem
            )
        return contract

    def _baseline_evaluation(
        self,
        baseline_workspace: dict[Path, str],
        run_log: RunLog,
    ) -> _Evaluation | None:
        if not self.target.has_contract:
            return None
        source = current_function(
            baseline_workspace[self.target.source_path.resolve()],
            self.target.address,
        )
        if source is None:
            return None
        assert self.target.symbol is not None
        assert self.target.prototype is not None
        candidate = Candidate(
            symbol=self.target.symbol,
            prototype=self.target.prototype,
            source=source,
        )
        output = source_safety_feedback(candidate)
        if output is None:
            score, output = self.repository.compare(
                self.target,
                self.config.build_timeout,
            )
        else:
            score = None
        run_log.write_baseline(output)
        return _Evaluation(
            candidate,
            self.target,
            dict(baseline_workspace),
            score,
            output,
        )

    def _prepare_resumed_candidate(
        self,
    ) -> tuple[Candidate, TargetSpec, list[str]]:
        assert self.initial_candidate is not None
        candidate, changes = normalize_resumed_candidate(
            self.initial_candidate,
            self.repository,
            self.target.address,
        )
        bound_source, binding_changes = bind_supporting_address_symbols(
            candidate.source,
            candidate.supporting_insertions,
        )
        if binding_changes:
            candidate = candidate.model_copy(update={"source": bound_source})
            changes.extend(binding_changes)
        if self.target.has_contract:
            if (
                candidate.symbol != self.target.symbol
                or candidate.prototype != self.target.prototype
            ):
                raise RuntimeError(
                    "resumed candidate contract does not match the established target"
                )
            return candidate, self.target, changes
        return candidate, self.target.with_candidate_contract(candidate), changes

    def _prepare_configured_seed(
        self,
        evidence: EvidenceBundle,
    ) -> tuple[Candidate, TargetSpec, list[str]]:
        """Build a Ghidra seed when the caller supplied a fixed contract."""

        assert self.target.symbol is not None
        assert self.target.prototype is not None
        contract = ContractProposal(
            symbol=self.target.symbol,
            prototype=self.target.prototype,
        )
        normalized, changes = normalize_decompiler_seed(
            evidence.decompiler_hint,
            self.repository,
            contract,
            excluded_address=self.target.address,
        )
        candidate = candidate_from_seed(
            normalized,
            contract,
            self.target.address,
        )
        return candidate, self.target, changes

    def _write_dry_run(
        self,
        run_log: RunLog,
        evidence: EvidenceBundle,
        evaluation: _Evaluation | None,
    ) -> SearchResult:
        if evaluation is None:
            prompt = build_contract_prompt(self.target, evidence)
            run_log.write_contract_prompt(prompt)
            stage = "contract"
            path = run_log.directory / "contract.prompt.txt"
        elif evaluation.score is not None:
            feedback = self.repository.compact_similarity_feedback(evaluation.output)
            prompt = build_similarity_patch_prompt(
                evaluation.target,
                evidence,
                evaluation.candidate,
                feedback,
            )
            run_log.write_edit_prompt(1, "similarity", prompt)
            stage = "similarity-edit"
            path = run_log.directory / "edit-01.similarity.prompt.txt"
        elif self.repository.is_repairable_build_failure(evaluation.output):
            feedback = self.repository.compact_compile_feedback(
                evaluation.output,
                evaluation.workspace.get(evaluation.target.source_path),
            )
            prompt = build_compile_patch_prompt(
                evaluation.target,
                evidence,
                evaluation.candidate,
                feedback,
            )
            run_log.write_edit_prompt(1, "compile", prompt)
            stage = "compile-repair"
            path = run_log.directory / "edit-01.compile.prompt.txt"
        else:
            run_log.update_manifest(
                status="failed",
                stage="build-or-compare",
                error="comparison failed without compiler diagnostics",
            )
            print("logs: %s" % run_log.directory)
            return SearchResult(
                score=None,
                target_reached=False,
                attempts=0,
                session_directory=run_log.directory,
                symbol=evaluation.target.symbol,
                prototype=evaluation.target.prototype,
            )
        run_log.update_manifest(status="dry-run", stage=stage)
        print(path)
        return SearchResult(
            score=evaluation.score if evaluation is not None else None,
            target_reached=False,
            attempts=0,
            session_directory=run_log.directory,
            symbol=evaluation.target.symbol if evaluation is not None else None,
            prototype=evaluation.target.prototype if evaluation is not None else None,
        )

    def run(self, dry_run_prompt: bool = False) -> SearchResult:
        evidence = self.repository.collect_evidence(
            self.target,
            self.config.max_callees,
        )
        baseline_workspace = self.repository.snapshot_workspace(self.target)
        output_root = self.repository.config.resolve(
            self.repository.root,
            self.repository.config.output_dir,
        )
        run_log = RunLog(output_root, self.target, self.config)
        reserved_symbols = set(self.repository.reserved_symbols(self.target))
        allowed_support_paths = self.repository.allowed_support_paths()

        best_workspace = dict(baseline_workspace)
        try:
            best_evaluation = self._baseline_evaluation(baseline_workspace, run_log)
        except BaseException as error:
            run_log.update_manifest(
                status="failed",
                stage="baseline",
                error={"type": type(error).__name__, "message": str(error)},
            )
            raise
        working = best_evaluation
        attempts = 0
        stop_reason = "edit-budget-exhausted"
        if working is not None:
            if working.score is not None:
                print("baseline similarity: %.2f%%" % working.score, flush=True)
                if working.score >= self.config.target_score:
                    result = SearchResult(
                        score=working.score,
                        target_reached=True,
                        attempts=0,
                        session_directory=run_log.directory,
                        symbol=working.target.symbol,
                        prototype=working.target.prototype,
                    )
                    run_log.update_manifest(
                        status="complete",
                        stop_reason="target-already-reached",
                        result={
                            **result.__dict__,
                            "session_directory": str(result.session_directory),
                        },
                    )
                    print("target already meets %.2f%%" % self.config.target_score)
                    print("logs: %s" % run_log.directory)
                    return result

        try:
            if self.initial_candidate is not None:
                candidate, active_target, changes = self._prepare_resumed_candidate()
                problem = _contract_problem(
                    ContractProposal(
                        symbol=candidate.symbol,
                        prototype=candidate.prototype,
                    ),
                    reserved_symbols,
                    self.target.address,
                )
                if problem is None:
                    working = self._evaluate(
                        candidate,
                        active_target,
                        baseline_workspace,
                        best_workspace,
                        reserved_symbols,
                        allowed_support_paths,
                    )
                    attempts += 1
                    run_log.write_seed(
                        candidate,
                        changes,
                        working.output,
                        working.score,
                        "resumed-candidate",
                    )
                    if working.score is not None and (
                        best_evaluation is None
                        or best_evaluation.score is None
                        or working.score > best_evaluation.score
                    ):
                        best_evaluation = working
                        best_workspace = working.workspace
                        self.repository.apply_workspace(best_workspace)

            if (
                working is None
                and self.initial_candidate is None
                and self.target.has_contract
            ):
                candidate, active_target, changes = self._prepare_configured_seed(
                    evidence
                )
                working = self._evaluate(
                    candidate,
                    active_target,
                    baseline_workspace,
                    best_workspace,
                    reserved_symbols,
                    allowed_support_paths,
                )
                attempts += 1
                run_log.write_seed(
                    candidate,
                    changes,
                    working.output,
                    working.score,
                    "configured-contract",
                )
                if working.score is not None and (
                    best_evaluation is None
                    or best_evaluation.score is None
                    or working.score > best_evaluation.score
                ):
                    best_evaluation = working
                    best_workspace = working.workspace
                    self.repository.apply_workspace(best_workspace)

            if dry_run_prompt:
                return self._write_dry_run(run_log, evidence, working)

            needs_contract = working is None
            if self.initial_candidate is not None:
                prepared, _, _ = self._prepare_resumed_candidate()
                problem = _contract_problem(
                    ContractProposal(
                        symbol=prepared.symbol,
                        prototype=prepared.prototype,
                    ),
                    reserved_symbols,
                    self.target.address,
                )
                needs_contract = problem is not None

            if (
                working is not None
                and working.score is not None
                and working.score >= self.config.target_score
            ):
                stop_reason = "target-reached-by-seed"
            elif self.config.max_edits == 0 and not needs_contract:
                stop_reason = "edits-disabled"
            else:
                server = ManagedLlamaServer(
                    self.server_config,
                    self.config,
                    run_log.directory,
                )
                with server:
                    run_log.update_manifest(
                        model_preset=self.server_config.resolved_preset().value
                    )
                    client = StructuredModelClient(server, self.config)
                    try:
                        if needs_contract:
                            if self.initial_candidate is not None:
                                candidate, _, changes = (
                                    self._prepare_resumed_candidate()
                                )
                                contract = ContractProposal(
                                    symbol=candidate.symbol,
                                    prototype=candidate.prototype,
                                )
                                origin = "resumed-candidate"
                            else:
                                prompt = build_contract_prompt(self.target, evidence)
                                run_log.write_contract_prompt(prompt)
                                print("requesting Qwen contract", flush=True)
                                contract, completion = client.infer_contract(prompt)
                                contract = normalize_contract(contract, self.repository)
                                run_log.write_contract_generation(contract, completion)
                                candidate = None
                                changes = []
                                origin = "ghidra-decompilation"

                            contract = self._repair_contract_name(
                                client,
                                run_log,
                                evidence,
                                contract,
                                reserved_symbols,
                            )
                            if candidate is not None:
                                if candidate.symbol != contract.symbol:
                                    candidate = rename_candidate_symbol(
                                        candidate,
                                        contract.symbol,
                                    )
                                if candidate.prototype != contract.prototype:
                                    candidate = candidate.model_copy(
                                        update={"prototype": contract.prototype}
                                    )
                                    candidate = candidate_from_seed(
                                        candidate.source,
                                        contract,
                                        self.target.address,
                                        candidate.supporting_insertions,
                                    )
                            else:
                                normalized, changes = normalize_decompiler_seed(
                                    evidence.decompiler_hint,
                                    self.repository,
                                    contract,
                                    excluded_address=self.target.address,
                                )
                                candidate = candidate_from_seed(
                                    normalized,
                                    contract,
                                    self.target.address,
                                )
                            active_target = self.target.with_candidate_contract(
                                candidate
                            )
                            working = self._evaluate(
                                candidate,
                                active_target,
                                baseline_workspace,
                                best_workspace,
                                reserved_symbols,
                                allowed_support_paths,
                            )
                            attempts += 1
                            run_log.write_seed(
                                candidate,
                                changes,
                                working.output,
                                working.score,
                                origin,
                            )
                            if working.score is not None and (
                                best_evaluation is None
                                or best_evaluation.score is None
                                or working.score > best_evaluation.score
                            ):
                                best_evaluation = working
                                best_workspace = working.workspace
                                self.repository.apply_workspace(best_workspace)
                            print(
                                "mechanical seed: %s"
                                % (
                                    "build failed"
                                    if working.score is None
                                    else "%.2f%%" % working.score
                                ),
                                flush=True,
                            )

                        if working is None:
                            raise RuntimeError("no reconstruction seed is available")

                        rejection_history: list[str] = []
                        for round_number in range(1, self.config.max_edits + 1):
                            if (
                                working.score is not None
                                and working.score >= self.config.target_score
                            ):
                                stop_reason = "target-reached"
                                break
                            if working.score is None:
                                if not self.repository.is_repairable_build_failure(
                                    working.output
                                ):
                                    stop_reason = "non-compiler-build-failure"
                                    break
                                kind = "compile"
                                feedback = self.repository.compact_compile_feedback(
                                    working.output,
                                    working.workspace.get(
                                        working.target.source_path,
                                    ),
                                )
                                prompt = build_compile_patch_prompt(
                                    working.target,
                                    evidence,
                                    working.candidate,
                                    feedback,
                                    "\n\n".join(rejection_history[-3:]),
                                )
                                run_log.write_edit_prompt(round_number, kind, prompt)
                                print(
                                    "edit %d/%d: requesting compile repair"
                                    % (round_number, self.config.max_edits),
                                    flush=True,
                                )
                                raw_patch, completion = client.repair_compile(
                                    prompt,
                                    round_number,
                                )
                            else:
                                kind = "similarity"
                                feedback = self.repository.compact_similarity_feedback(
                                    working.output
                                )
                                prompt = build_similarity_patch_prompt(
                                    working.target,
                                    evidence,
                                    working.candidate,
                                    feedback,
                                    "\n\n".join(rejection_history[-3:]),
                                )
                                run_log.write_edit_prompt(round_number, kind, prompt)
                                print(
                                    "edit %d/%d: requesting similarity edit"
                                    % (round_number, self.config.max_edits),
                                    flush=True,
                                )
                                proposed, completion = client.improve_similarity(
                                    prompt,
                                    round_number,
                                )
                                assert isinstance(proposed, SimilarityPatch)
                                raw_patch = SourcePatch(edits=[proposed.edit])

                            run_log.write_edit_generation(
                                round_number,
                                kind,
                                raw_patch,
                                completion,
                            )
                            patch, rejected = sanitize_source_patch(
                                working.candidate.source,
                                raw_patch,
                                working.candidate.symbol,
                            )
                            if (
                                not patch.edits
                                and not patch.identifier_replacements
                                and not patch.supporting_insertions
                            ):
                                rejection = (
                                    "Keep CURRENT SOURCE unchanged. Attempted patch: %s\n%s"
                                    % (
                                        raw_patch.model_dump_json(),
                                        "; ".join(rejected)
                                        or "the model returned an empty patch",
                                    )
                                )
                                rejection_history.append(rejection)
                                run_log.write_edit_result(
                                    round_number,
                                    kind,
                                    None,
                                    patch,
                                    rejected,
                                    "PATCH NOT APPLIED: " + rejection,
                                    None,
                                    False,
                                    {},
                                )
                                print("edit %d: rejected before build" % round_number)
                                continue

                            try:
                                trial_candidate = apply_source_patch(
                                    working.candidate,
                                    patch,
                                    self.target.address,
                                )
                                trial = self._evaluate(
                                    trial_candidate,
                                    working.target,
                                    baseline_workspace,
                                    best_workspace,
                                    reserved_symbols,
                                    allowed_support_paths,
                                )
                                attempts += 1
                            except ValueError as error:
                                rejection = (
                                    "Keep CURRENT SOURCE unchanged. Attempted patch: %s\n"
                                    "Candidate rejected before measurement: %s"
                                    % (patch.model_dump_json(), error)
                                )
                                rejection_history.append(rejection)
                                run_log.write_edit_result(
                                    round_number,
                                    kind,
                                    None,
                                    patch,
                                    rejected,
                                    rejection,
                                    None,
                                    False,
                                    {},
                                )
                                print("edit %d: rejected before build" % round_number)
                                continue

                            run_log.write_edit_result(
                                round_number,
                                kind,
                                trial.candidate,
                                patch,
                                rejected,
                                trial.output,
                                trial.score,
                                True,
                                patch_metrics(trial.candidate, patch),
                            )
                            working = trial
                            rejection_history.clear()
                            if trial.score is not None and (
                                best_evaluation is None
                                or best_evaluation.score is None
                                or trial.score > best_evaluation.score
                            ):
                                best_evaluation = trial
                                best_workspace = trial.workspace
                                self.repository.apply_workspace(best_workspace)
                            print(
                                "edit %d: followed, %s"
                                % (
                                    round_number,
                                    (
                                        "source safety still failing"
                                        if self.repository.has_source_safety_errors(
                                            trial.output
                                        )
                                        else (
                                            "build still failing"
                                            if self.repository.has_compiler_errors(
                                                trial.output
                                            )
                                            else "comparison failed"
                                        )
                                    )
                                    if trial.score is None
                                    else "similarity %.2f%%" % trial.score,
                                ),
                                flush=True,
                            )
                        else:
                            stop_reason = "edit-budget-exhausted"
                    except ModelRequestError as error:
                        if best_evaluation is None or best_evaluation.score is None:
                            raise
                        stop_reason = "model-request-failed-after-compiled-seed"
                        run_log.update_manifest(model_error=str(error))
                        print(
                            "model request failed; preserving the best compiled seed",
                            flush=True,
                        )
                    finally:
                        client.close()

            score = best_evaluation.score if best_evaluation is not None else None
            target_reached = score is not None and score >= self.config.target_score
            if target_reached and stop_reason == "edit-budget-exhausted":
                stop_reason = "target-reached"
            selected_target = (
                best_evaluation.target
                if best_evaluation is not None
                else (working.target if working is not None else self.target)
            )
            changed_files: list[str] = []
            if score is not None and best_evaluation is not None:
                changed_files = self.repository.changed_workspace_files(
                    baseline_workspace,
                    best_workspace,
                )
                run_log.write_selected(
                    best_evaluation.candidate,
                    score,
                    changed_files,
                )
            result = SearchResult(
                score=score,
                target_reached=target_reached,
                attempts=attempts,
                session_directory=run_log.directory,
                symbol=selected_target.symbol,
                prototype=selected_target.prototype,
            )
            run_log.update_manifest(
                status="complete" if score is not None else "failed",
                stop_reason=stop_reason,
                result={
                    **result.__dict__,
                    "session_directory": str(result.session_directory),
                    "changed_files": changed_files,
                },
            )
            if score is None:
                print("no candidate compiled; original source restored", flush=True)
            else:
                print("best retained similarity: %.2f%%" % score, flush=True)
                print(
                    "selected contract: %s — %s"
                    % (selected_target.symbol, selected_target.prototype),
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

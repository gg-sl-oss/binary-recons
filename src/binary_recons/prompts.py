"""Prompt construction for deterministic candidate batches."""

from __future__ import annotations

from dataclasses import dataclass

from .models import EvidenceBundle, SearchConfig, TargetSpec


SYSTEM_PROMPT = """\
You are searching for the original source form of one compiled function.
Original assembly is authoritative and decompiled source is only a semantic
hint. Infer changes yourself from mechanically supplied compiler comparisons.
Populate only the requested structured response; do not return analysis,
Markdown, patches, or unrelated edits.
"""


@dataclass(frozen=True)
class HistoryItem:
    fingerprint: str
    score: float | None
    candidate: str


def _format_history(history: list[HistoryItem], limit: int) -> str:
    if not history or limit == 0:
        return "No prior candidates."
    blocks: list[str] = []
    for item in history[-limit:]:
        score = "build failed" if item.score is None else "%.2f%%" % item.score
        blocks.append(
            "fingerprint %s, score %s\n%s"
            % (item.fingerprint[:12], score, item.candidate)
        )
    return "\n\n".join(blocks)


def build_prompt(
    target: TargetSpec,
    evidence: EvidenceBundle,
    config: SearchConfig,
    best_candidate: str | None,
    feedback: str,
    history: list[HistoryItem],
) -> str:
    current = best_candidate if best_candidate is not None else "<not implemented>"
    last_result = feedback if feedback else "No candidate has been compiled yet."
    roles = [
        "- Candidate 1 must isolate a declaration/order/lifetime hypothesis. "
        "When values occupy different registers or stack slots, vary only meaningful "
        "parameter copies or locals supported by the semantics.",
        "- Candidate 2 must isolate a control-flow or expression-order hypothesis. "
        "Use pre/post side effects when the original copies a value before changing "
        "it but the rebuild folds the operation into an indexed access.",
        "- Candidate 3 must combine the strongest evidence-supported hypotheses for "
        "the currently unmatched regions instead of making cosmetic rewrites.",
    ]
    role_text = "\n".join(roles[: config.candidates_per_iteration])
    if config.candidates_per_iteration > len(roles):
        role_text += (
            "\n- Additional candidates must explore other independent, legitimate "
            "C source-form axes visible in the compiler comparison."
        )

    return f"""\
Produce exactly {config.candidates_per_iteration} materially different complete
source definitions in the structured candidates field.

TARGET CONTRACT
- Marker: /* Function start: 0x{target.address:X} */
- Signature: {target.prototype}
- Define only {target.symbol}; include no declarations or surrounding file text.

TARGET PROJECT
- Language: {evidence.language}
- Compiler/toolchain: {evidence.compiler}
- Original assembly below is the only authority. Decompiled C is a hint.
- Preserve the exact signature and calling convention.
- Define only the target function, with no includes, declarations, helpers,
  surrounding file text, Markdown, placeholders, or unrelated edits.
- Use only identifiers supported by the supplied project/declaration evidence.

TARGET-PROJECT RULES
{evidence.project_guidance}

GENERIC SOURCE-FORM SEARCH LEVERS
- Infer types, signedness, declaration order, value lifetimes, expression
  grouping, casts, and whether a parameter or meaningful local carries state.
- Try source-level loop/branch equivalents, fall-through direction, early vs
  common returns, and pre/post increment forms when instruction order asks.
- Treat register choice, stack cleanup, operand width, and jump direction as
  evidence. Comparisons show rebuilt instructions LEFT and original RIGHT.
- Prefer exact identifiers from declaration evidence. An instruction displacement
  may be a global base plus a member offset; do not assume it names a standalone
  object.
- Do not repeat a prior fingerprint and do not explain an inference.

BATCH EXPERIMENT DESIGN
{role_text}

ORIGINAL TARGET ASSEMBLY
{evidence.original_assembly}

DECOMPILER HINT
{evidence.decompiler_hint}

MECHANICALLY DISCOVERED REFERENCED STRINGS
{evidence.string_evidence}

MECHANICALLY DISCOVERED DIRECT-CALLEE EVIDENCE
{evidence.callee_evidence}

MECHANICALLY DISCOVERED REFERENCED DECLARATIONS
{evidence.declaration_evidence}

CURRENT BEST CANDIDATE
{current}

LAST COMPILER / BINARY-COMP RESULT
{last_result}

RECENT CANDIDATES ALREADY TRIED
{_format_history(history, config.history_limit)}
"""


def build_compile_repair_prompt(
    target: TargetSpec,
    evidence: EvidenceBundle,
    config: SearchConfig,
    candidate: str,
    compiler_feedback: str,
    repair_attempt: int,
) -> str:
    return f"""\
Repair the failing source definition below. Produce exactly one complete corrected
definition in the structured source field.

REPAIR PASS
- Attempt {repair_attempt} of {config.compile_repair_attempts}.
- This is a narrow compilation repair, not a new reconstruction attempt.
- Preserve the candidate's behavior, control-flow shape, meaningful locals,
  signature, marker, and function name unless a compiler diagnostic requires a
  source-level correction.
- Fix every compiler diagnostic shown. Do not explain the fix.

TARGET CONTRACT
- Marker: /* Function start: 0x{target.address:X} */
- Signature: {target.prototype}
- Define only {target.symbol}; include no declarations or surrounding file text.

COMPILATION CONSTRAINTS
- Language: {evidence.language}
- Compiler/toolchain: {evidence.compiler}
- Use exact identifiers and types from the supplied declaration evidence.
- Do not invent helpers, wrappers, globals, fields, types, macros, or includes.
- No placeholders, omitted bodies, Markdown, or unrelated edits.

TARGET-PROJECT RULES
{evidence.project_guidance}

COMPILER DIAGNOSTICS
{compiler_feedback}

MECHANICALLY DISCOVERED REFERENCED DECLARATIONS
{evidence.declaration_evidence}

FAILING C DEFINITION
{candidate}
"""

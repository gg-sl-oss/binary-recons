"""Prompt construction for deterministic candidate batches."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Candidate, EvidenceBundle, SearchConfig, TargetSpec


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
    best_candidate: Candidate | None,
    feedback: str,
    history: list[HistoryItem],
) -> str:
    current = (
        best_candidate.model_dump_json(indent=2)
        if best_candidate is not None
        else "<not implemented>"
    )
    last_result = feedback if feedback else "No candidate has been compiled yet."
    if target.has_contract:
        contract = f"""\
- Existing symbol: {target.symbol}
- Existing signature: {target.prototype}
- Preserve that symbol, signature, and calling convention in every candidate."""
    else:
        contract = """\
- No source-level name or interface is supplied for this unimplemented target.
- Infer a concise, meaningful function name and complete prototype from the raw
  assembly, decompiler hint, callers/callees, strings, and declarations.
- Do not use a Ghidra label, address, generic placeholder, or mechanism-only
  name. Each candidate's symbol and prototype fields must exactly describe its
  source definition."""
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
{contract}
- Define only the proposed target; include no declarations or surrounding file
  text.

TARGET PROJECT
- Language: {evidence.language}
- Compiler/toolchain: {evidence.compiler}
- Original assembly below is the only authority. Decompiled C is a hint.
- Return structured symbol, prototype, and source fields for every candidate.
- Define only the target function, with no includes, declarations, helpers,
  surrounding file text, Markdown, placeholders, or unrelated edits.
- Use only identifiers supported by the target or supplied project/declaration
  evidence.
- Return any target-required new types, globals, or matching global definitions
  in supporting_insertions. Each insertion path must exactly match an allowed
  support file below. These are append-only blocks: never repeat existing file
  contents, edit existing declarations, or place the target prototype there.
- A candidate must be a complete change set. If its function uses a new type or
  global, include every declaration and definition needed to compile it. Use an
  empty supporting_insertions list when no support is needed.

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

RESERVED EXISTING SOURCE SYMBOLS
{evidence.reserved_symbols}

MECHANICALLY DISCOVERED DIRECT-CALLEE EVIDENCE
{evidence.callee_evidence}

MECHANICALLY DISCOVERED REFERENCED DECLARATIONS
{evidence.declaration_evidence}

ALLOWED SUPPORT FILES AND CURRENT CONTENT
{evidence.supporting_file_evidence}

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
    candidate: Candidate,
    compiler_feedback: str,
    repair_attempt: int,
) -> str:
    return f"""\
Repair the failing change set below. Produce exactly one complete corrected
structured change set.

REPAIR PASS
- Attempt {repair_attempt} of {config.compile_repair_attempts}.
- This is a narrow compilation repair, not a new reconstruction attempt.
- Preserve the candidate's behavior, control-flow shape, meaningful locals,
  signature, marker, function name, and valid supporting insertions unless a
  build diagnostic requires a source-level correction.
- Fix every compiler diagnostic shown. Do not explain the fix.

TARGET CONTRACT
- Marker: /* Function start: 0x{target.address:X} */
- Symbol: {target.symbol}
- Signature: {target.prototype}
- Restore and preserve this active name and interface if the failing candidate
  changed either one.
- Return symbol, prototype, and source fields that agree exactly.
- Define only the target; include no declarations or surrounding file text.
- Return a complete repaired supporting_insertions list. It replaces, rather
  than supplements, the failing list.

COMPILATION CONSTRAINTS
- Language: {evidence.language}
- Compiler/toolchain: {evidence.compiler}
- Use exact identifiers and types from the supplied declaration evidence.
- Add a type or global through an allowed support file only when target evidence
  or a build diagnostic requires it. Do not invent unsupported helpers,
  wrappers, fields, macros, or includes.
- No placeholders, omitted bodies, Markdown, or unrelated edits.

TARGET-PROJECT RULES
{evidence.project_guidance}

BUILD / VALIDATION DIAGNOSTICS
{compiler_feedback}

MECHANICALLY DISCOVERED REFERENCED DECLARATIONS
{evidence.declaration_evidence}

ALLOWED SUPPORT FILES AND CURRENT CONTENT
{evidence.supporting_file_evidence}

FAILING COMPLETE CHANGE SET
{candidate.model_dump_json(indent=2)}
"""


def build_validation_repair_prompt(
    target: TargetSpec,
    evidence: EvidenceBundle,
    config: SearchConfig,
    candidate: Candidate,
    validation_feedback: str,
    repair_attempt: int,
) -> str:
    if target.has_contract:
        contract = f"""\
- Active symbol: {target.symbol}
- Active signature: {target.prototype}
- The target contract is established. Preserve it exactly."""
    else:
        contract = """\
- This target had no established source-level contract.
- You may replace the proposed symbol and prototype when the validation failure
  requires it. Keep symbol, prototype, and source definition mutually exact.
- A replacement name must remain concise, meaningful, and grounded in the
  target behavior; never use an address or generic operational label."""

    return f"""\
Repair the rejected change set below. Produce exactly one complete corrected
structured change set.

REPAIR PASS
- Attempt {repair_attempt} of {config.compile_repair_attempts}.
- This is a narrow pre-build validation repair, not a new reconstruction pass.
- Preserve the implementation's behavior and source shape except where the
  validation diagnostic requires a correction.
- Fix every validation diagnostic shown. Do not explain the fix.

TARGET CONTRACT
- Marker: /* Function start: 0x{target.address:X} */
{contract}
- Return a complete supporting_insertions list replacing the failing list.

TARGET-PROJECT RULES
{evidence.project_guidance}

VALIDATION DIAGNOSTICS
{validation_feedback}

RESERVED EXISTING SOURCE SYMBOLS
{evidence.reserved_symbols}

DECOMPILER HINT
{evidence.decompiler_hint}

MECHANICALLY DISCOVERED REFERENCED DECLARATIONS
{evidence.declaration_evidence}

ALLOWED SUPPORT FILES AND CURRENT CONTENT
{evidence.supporting_file_evidence}

REJECTED COMPLETE CHANGE SET
{candidate.model_dump_json(indent=2)}
"""

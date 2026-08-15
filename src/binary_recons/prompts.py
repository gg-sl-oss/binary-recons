"""Prompt construction for deterministic candidate batches."""

from __future__ import annotations

from dataclasses import dataclass

from .models import EvidenceBundle, SearchConfig, TargetSpec


SYSTEM_PROMPT = """\
You are searching for the original C source form of one function compiled by
Microsoft Visual C++ 4.20. Original assembly is authoritative and decompiled C
is only a semantic hint. Infer changes yourself from mechanically supplied
compiler comparisons. Populate only the requested structured candidate batch;
do not return analysis, Markdown, patches, or unrelated edits.
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
    think_control = "/no_think\n" if not config.thinking else ""
    current = best_candidate if best_candidate is not None else "<not implemented>"
    last_result = feedback if feedback else "No candidate has been compiled yet."
    roles = [
        "- Candidate 1 must isolate a declaration/order/lifetime hypothesis. "
        "When aligned values occupy different registers, vary only meaningful "
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
{think_control}Produce exactly {config.candidates_per_iteration} materially different complete C
definitions in the structured candidates field.

TARGET CONTRACT
- Marker: /* Function start: 0x{target.address:X} */
- Signature: {target.prototype}
- Define only {target.symbol}; include no declarations or surrounding file text.

PROJECT CONSTRAINTS
- Microsoft Visual C++ 4.20 optimized C; game core is C, not C++.
- Original assembly below is the only authority. Decompiled C is a hint.
- Preserve the exact signature and calling convention.
- Expect DOS-port 16-bit types; infer widths and signedness from instructions.
- No helpers, wrappers, dummy variables, inline assembly, exception handling,
  unions/substructures, includes, declarations, or unrelated edits.
- Prefer known struct fields over raw pointer arithmetic, but a useful first-pass
  hypothesis is better than refusing to produce code.
- Keep one C statement per line and preserve the meaningful function name.

GENERIC SOURCE-FORM SEARCH LEVERS
- Infer types, signedness, declaration order, value lifetimes, expression
  grouping, casts, and whether a parameter or meaningful local carries state.
- Try source-level loop/branch equivalents, fall-through direction, early vs
  common returns, and pre/post increment forms when instruction order asks.
- Treat register choice, stack cleanup, operand width, and jump direction as
  evidence. Comparisons show rebuilt instructions LEFT and original RIGHT.
- Use exact central identifiers from declaration evidence. An instruction
  displacement may be a global base plus a member offset; never invent a symbol
  by copying that effective address into its name.
- Replace decompiler aliases with supplied address-matched central identifiers;
  never emit an identifier that the evidence reports as undeclared.
- Do not repeat a prior fingerprint and do not explain an inference.

BATCH EXPERIMENT DESIGN
{role_text}

ORIGINAL TARGET ASSEMBLY
{evidence.original_assembly}

DECOMPILER HINT
{evidence.decompiler_hint}

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

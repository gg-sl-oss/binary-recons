"""Small, stage-specific prompts for the compile-first reconstruction loop."""

from __future__ import annotations

from .models import Candidate, ContractProposal, EvidenceBundle, TargetSpec


SYSTEM_PROMPT = """\
You assist a deterministic binary source-reconstruction driver. Return only the
requested structured data. Never return analysis, Markdown, a whole replacement
function, shell commands, tool calls, or unrelated edits. Original assembly and
compiler feedback are authoritative; decompiled C is only a semantic seed.
"""


def _excerpt(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... prompt context mechanically truncated ..."


def build_contract_prompt(target: TargetSpec, evidence: EvidenceBundle) -> str:
    """Ask Qwen for the only non-mechanical bootstrap decision."""

    return f"""\
TASK
Infer only a meaningful source-level function name and complete prototype for
the target below. Return CONTRACT DATA ONLY. Do not reconstruct or edit the
body. No target name or interface has been supplied by the driver.

CONTRACT REQUIREMENTS
- The decompiler's FUN/sub/operational label is forbidden as a source name.
- Use a concise purpose-based VerbObject name, without an address and without a
  vague Helper/Handler/Function/Procedure label.
- Include parameter names for every non-void parameter so the driver can align
  the decompiler body mechanically.
- Preserve parameter order, return type, pointer depth, language, and calling
  convention supported by the evidence. Do not invent a class or wrapper.
- Use types accepted by the configured historical compiler.

TARGET
- Address (evidence key only): 0x{target.address:08X}
- Language: {evidence.language}
- Compiler/toolchain: {evidence.compiler}

GHIDRA DECOMPILATION (BEHAVIOR AND INTERFACE HINT)
{_excerpt(evidence.decompiler_hint)}

MECHANICALLY DISCOVERED REFERENCED STRINGS
{_excerpt(evidence.string_evidence, 1500)}

DIRECT-CALLEE EVIDENCE
{_excerpt(evidence.callee_evidence, 3000)}

RELEVANT EXISTING DECLARATIONS
{_excerpt(evidence.declaration_evidence, 4000)}

RESERVED EXISTING FUNCTION NAMES
{_excerpt(evidence.reserved_symbols, 2500)}

PROJECT AND COMPILER RULES
{_excerpt(evidence.project_guidance, 3500)}

Return only the structured contract.
"""


def build_symbol_repair_prompt(
    target: TargetSpec,
    evidence: EvidenceBundle,
    contract: ContractProposal,
    reason: str,
) -> str:
    """Repair only a colliding or operational contract name."""

    return f"""\
NAMING VALIDATION FAILURE
{reason}

TASK
Return exactly six fresh, distinct, meaningful C function identifiers, strongest
first. This is a naming task only. Keep the interface concept locked, do not
reconstruct the body, and do not restate or alter the prototype.

LOCKED INTERFACE
{contract.prototype}

REQUIREMENTS
- Infer purpose from behavior, strings, and calls.
- Diversify the leading verbs and object nouns across the choices.
- Do not return `{contract.symbol}`, an address, a FUN/sub label, a reserved
  name, or a vague Helper/Handler/Function/Procedure name.

TARGET ADDRESS (EVIDENCE KEY ONLY)
0x{target.address:08X}

GHIDRA DECOMPILATION
{_excerpt(evidence.decompiler_hint)}

REFERENCED STRINGS
{evidence.string_evidence}

RESERVED NAMES
{evidence.reserved_symbols}

Return only the structured name list.
"""


def build_compile_patch_prompt(
    target: TargetSpec,
    evidence: EvidenceBundle,
    candidate: Candidate,
    compiler_feedback: str,
    previous_rejection: str = "",
) -> str:
    """Put the failing compiler output first so a weak model sees the root cause."""

    rejection = previous_rejection or "No earlier patch was rejected."
    return f"""\
COMPILER FEEDBACK FOR THE CURRENT DRAFT
{compiler_feedback.strip()}

PREVIOUS PATCH REJECTION
{_excerpt(rejection, 3500)}

TASK
Fix only the first compiler root cause with the smallest possible source patch.
This is a COMPILE-ONLY pass: do not inspect or tune assembly, rename the target,
change its prototype, regenerate the function, or edit another file. Return
SOURCE PATCH DATA ONLY.

PATCH CONTRACT
- The driver accepts at most one exact edit plus up to eight whole-token
  identifier replacements.
- Exact `old` text must occur verbatim. Use mode `once` unless every occurrence
  is independently wrong.
- Use identifier replacements only for simple existing-token renames.
- Pointer arithmetic and imperfect types may remain when they compile.
- The exact first blocking source line is supplied above. Patch that line or a
  token used by it; do not spend this turn cleaning up an unrelated warning.
- Prefer supplied declarations. If an unresolved DAT/global has no declaration,
  replace that identifier with a short typed absolute-address lvalue expression.
- Do not add behavior, a helper, a declaration, an include, a macro, Markdown,
  or an operational FUN_/DAT_ identifier.

LOCKED FUNCTION CONTRACT
{candidate.prototype}

CURRENT SOURCE
{candidate.source}

GHIDRA DECOMPILATION (BEHAVIOR HINT)
{_excerpt(evidence.decompiler_hint)}

RELEVANT EXISTING DECLARATIONS
{_excerpt(evidence.declaration_evidence, 5000)}

DIRECT-CALLEE EVIDENCE
{_excerpt(evidence.callee_evidence, 3500)}

PROJECT AND COMPILER RULES
{_excerpt(evidence.project_guidance, 3500)}

Target address 0x{target.address:08X} is for evidence selection only. Return only
the structured source patch.
"""


def build_similarity_patch_prompt(
    target: TargetSpec,
    evidence: EvidenceBundle,
    candidate: Candidate,
    comparison_feedback: str,
    previous_rejection: str = "",
) -> str:
    """Ask for one evidence-backed source experiment against the current diff."""

    rejection = previous_rejection or "No earlier patch was rejected."
    return f"""\
LATEST BINARY-COMP FEEDBACK (AUTHORITATIVE)
{comparison_feedback.strip()}

DIFF ORIENTATION
Each comparison row is `current compiler output | original binary`. Relocated
instruction and data addresses are not source mismatches.

PREVIOUS PATCH REJECTION
{_excerpt(rejection, 3500)}

TASK
The function compiles. Improve only the earliest important instruction-shape,
control-flow, operand-width, register-lifetime, or stack-offset mismatch with ONE
small exact source edit. Return PATCH DATA ONLY. Never regenerate the function,
rename it, change its prototype, add a helper/dummy variable, or edit another
file. Ignore relocated code and data addresses. Prefer a missing or extra source
operation evidenced by the decompilation and assembly—such as an index stride,
arithmetic scale, cast, temporary, or branch shape—over a semantic-only rename.

PATCH CONTRACT
- The exact `old` text must occur verbatim in the current source.
- Preserve brace balance, the address marker, and unrelated statements.
- Use mode `once` unless all occurrences are independently evidenced as wrong.
- A rejected patch is blacklisted. Never repeat its `old`/`new` edit; choose a
  different source region or a materially different source shape.
- Do not spend a turn on a global/field rename or constant-address substitution
  when it leaves the compiler's instruction shape unchanged.
- The driver will compile the trial transactionally and reject it unless its
  measured similarity strictly increases.

LOCKED FUNCTION CONTRACT
{candidate.prototype}

CURRENT SOURCE
{candidate.source}

ORIGINAL FUNCTION ASSEMBLY (AUTHORITATIVE)
{_excerpt(evidence.original_assembly, 9000)}

GHIDRA DECOMPILATION (SEMANTIC HINT, NOT AUTHORITATIVE)
{_excerpt(evidence.decompiler_hint, 7000)}

RELEVANT EXISTING DECLARATIONS
{_excerpt(evidence.declaration_evidence, 5000)}

DIRECT-CALLEE EVIDENCE
{_excerpt(evidence.callee_evidence, 2500)}

MECHANICALLY DISCOVERED REFERENCED STRINGS
{_excerpt(evidence.string_evidence, 1000)}

COMPILER/TOOLCHAIN
{evidence.compiler}

PROJECT AND COMPILER RULES
{_excerpt(evidence.project_guidance, 3500)}

Target address 0x{target.address:08X} is for evidence selection only. Return only
the structured single edit.
"""

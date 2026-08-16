"""Small, stage-specific prompts for the compile-first reconstruction loop."""

from __future__ import annotations

from .models import Candidate, ContractProposal, EvidenceBundle, TargetSpec


SYSTEM_PROMPT = """\
You assist a deterministic binary source-reconstruction driver. Return only the
requested structured data. Never return analysis, Markdown, a whole replacement
function, shell commands, tool calls, or unrelated edits. Original assembly and
compiler feedback are authoritative; decompiled C is only a semantic seed.
Never turn a decompiler global into a cast of a numeric absolute address. Source
output must use meaningful declared globals, aggregate elements, or fields.
"""


def _excerpt(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... prompt context mechanically truncated ..."


def _pending_support(candidate: Candidate) -> str:
    if not candidate.supporting_insertions:
        return "No supporting declarations or definitions are pending."
    return "\n\n".join(
        "[%s]\n%s" % (insertion.path, insertion.content)
        for insertion in candidate.supporting_insertions
    )


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
    """Put blocking build or source-safety feedback before the draft."""

    rejection = (
        previous_rejection or "No earlier patch was rejected before measurement."
    )
    return f"""\
BUILD OR SOURCE-SAFETY FEEDBACK FOR THE CURRENT DRAFT
{compiler_feedback.strip()}

PREVIOUS INVALID PATCH
{_excerpt(rejection, 3500)}

TASK
Fix only the first blocking compiler or source-safety root cause with the
smallest possible source patch. This is a COMPILE/SAFETY pass: do not inspect or
tune assembly, rename the target, change its prototype, or regenerate the
function. Return SOURCE PATCH DATA ONLY.

PATCH CONTRACT
- The driver accepts at most one exact edit plus up to eight whole-token
  identifier replacements and three configured supporting insertions.
- Exact `old` text must occur verbatim. Use mode `once` unless every occurrence
  is independently wrong.
- Use identifier replacements only for simple existing-token renames.
- Pointer arithmetic and imperfect types may remain when they compile.
- The exact first blocking source line is supplied above. Patch that line or a
  token used by it; do not spend this turn cleaning up an unrelated warning.
- A numeric address cast to a pointer is always forbidden, even when it compiles
  or produces matching assembly. Never emit `*(type *)0x...`, `(type *)0x...`,
  or an address-based pointer-arithmetic variant.
- Replace unresolved DAT/PTR/UNK globals with meaningful source-level globals,
  aggregate elements, or fields. Prefer a compatible supplied declaration. An
  address inside a declared array/structure must be expressed through that
  aggregate, not as a new pointer.
- If the evidence requires a genuinely undeclared standalone global, use
  `supporting_insertions` to add its meaningful extern declaration and matching
  definition to the explicitly configured global files. Follow the project's
  naming/address convention. Never invent a generic `unknown`, `data`, or
  `value` name merely to compile.
- When a paired declaration preserves its address in the meaningful symbol,
  the driver mechanically propagates that model-chosen name to unambiguous
  DAT/PTR/UNK uses and fixed-width array elements. Choose the correct type,
  base address, and array length; do not spend separate turns renaming each use.
- `supporting_insertions` are never a way to add a local or edit the target
  source file. For a C89 `for (int i = ...)` error, use the exact source edit to
  declare `i` with the function's locals; a later repair can remove `int` from
  the loop initializer if both locations do not fit in one exact edit.
- When one unsafe address repeats, repair every occurrence that fits the same
  evidenced source-level object in this patch.
- Do not add behavior, a helper, an include, a macro, Markdown, or an operational
  FUN_/DAT_/PTR_/UNK_ identifier.

LOCKED FUNCTION CONTRACT
{candidate.prototype}

CURRENT SOURCE
{candidate.source}

GHIDRA DECOMPILATION (BEHAVIOR HINT)
{_excerpt(evidence.decompiler_hint)}

RELEVANT EXISTING DECLARATIONS
{_excerpt(evidence.declaration_evidence, 5000)}

CONFIGURED SUPPORT FILES
{_excerpt(evidence.supporting_file_evidence, 7000)}

CURRENT PENDING SUPPORT INSERTIONS
{_excerpt(_pending_support(candidate), 4000)}

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

    rejection = (
        previous_rejection or "No earlier patch was rejected before measurement."
    )
    return f"""\
LATEST BINARY-COMP FEEDBACK (AUTHORITATIVE)
{comparison_feedback.strip()}

DIFF ORIENTATION
Each comparison row is `current compiler output | original binary`. Relocated
instruction and data addresses are not source mismatches.

PREVIOUS INVALID PATCH
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
- A patch rejected before measurement is blacklisted. Never repeat its
  `old`/`new` edit; choose a different source region or source shape.
- Do not spend a turn on a global/field rename or constant-address substitution
  when it leaves the compiler's instruction shape unchanged.
- Never introduce an absolute-address pointer expression. Keep every memory
  access expressed through meaningful declared globals, aggregate elements, or
  fields.
- The driver follows every valid measured edit as the next working source,
  while retaining the best compiling candidate separately for final output.

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

CURRENT PENDING SUPPORT INSERTIONS
{_excerpt(_pending_support(candidate), 3000)}

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

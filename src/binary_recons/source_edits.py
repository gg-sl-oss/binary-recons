"""Validation and application of the model's bounded exact source edits."""

from __future__ import annotations

import re

from .models import Candidate, ContractProposal, ExactEdit, SourcePatch
from .seed import candidate_from_seed


_OPERATIONAL_IDENTIFIER_RE = re.compile(r"\b(?:DAT|FUN)_[0-9A-Fa-f]{8}\b")


def _unsafe_new_text(text: str) -> str | None:
    if _OPERATIONAL_IDENTIFIER_RE.search(text):
        return "reintroduces an operational decompiler identifier"
    if "Function start:" in text:
        return "changes the driver-managed function marker"
    if "```" in text or re.search(r"(?m)^\s*#", text):
        return "introduces Markdown or a preprocessor directive"
    return None


def sanitize_source_patch(
    source: str,
    patch: SourcePatch,
    locked_symbol: str,
) -> tuple[SourcePatch, list[str]]:
    """Retain independent safe operations when Qwen quotes stale source text."""

    working = source
    edits: list[ExactEdit] = []
    replacements = []
    rejected: list[str] = []

    for index, edit in enumerate(patch.edits, 1):
        unsafe = _unsafe_new_text(edit.new)
        if unsafe is not None:
            rejected.append("edit %d rejected: %s" % (index, unsafe))
            continue
        old_balance = edit.old.count("{") - edit.old.count("}")
        new_balance = edit.new.count("{") - edit.new.count("}")
        if old_balance != new_balance:
            rejected.append("edit %d rejected: changes brace balance" % index)
            continue
        count = working.count(edit.old)
        valid = count >= 1 if edit.mode == "all" else count == 1
        if not valid:
            expected = "at least one" if edit.mode == "all" else "one"
            rejected.append(
                "edit %d rejected: expected %s old text occurrence(s), found %d"
                % (index, expected, count)
            )
            continue
        working = (
            working.replace(edit.old, edit.new)
            if edit.mode == "all"
            else working.replace(edit.old, edit.new, 1)
        )
        edits.append(edit)

    for index, replacement in enumerate(patch.identifier_replacements, 1):
        if replacement.old == locked_symbol or replacement.new == locked_symbol:
            rejected.append(
                "identifier replacement %d rejected: changes the locked contract"
                % index
            )
            continue
        if _OPERATIONAL_IDENTIFIER_RE.fullmatch(replacement.new):
            rejected.append(
                "identifier replacement %d rejected: introduces an operational name"
                % index
            )
            continue
        pattern = r"\b%s\b" % re.escape(replacement.old)
        working, count = re.subn(pattern, replacement.new, working)
        if count == 0:
            rejected.append(
                "identifier replacement %d rejected: old token absent" % index
            )
            continue
        replacements.append(replacement)

    return (
        SourcePatch(identifier_replacements=replacements, edits=edits),
        rejected,
    )


def apply_source_patch(
    candidate: Candidate, patch: SourcePatch, address: int
) -> Candidate:
    """Apply a sanitized patch and mechanically reassert the locked definition header."""

    source = candidate.source
    for index, edit in enumerate(patch.edits, 1):
        count = source.count(edit.old)
        if edit.mode == "once" and count != 1:
            raise ValueError(
                "edit %d expected one occurrence, found %d" % (index, count)
            )
        if edit.mode == "all" and count == 0:
            raise ValueError("edit %d old text does not occur" % index)
        source = (
            source.replace(edit.old, edit.new, 1)
            if edit.mode == "once"
            else source.replace(edit.old, edit.new)
        )
    for index, replacement in enumerate(patch.identifier_replacements, 1):
        source, count = re.subn(
            r"\b%s\b" % re.escape(replacement.old),
            replacement.new,
            source,
        )
        if count == 0:
            raise ValueError(
                "identifier replacement %d old token does not occur" % index
            )
    return candidate_from_seed(
        source,
        contract=_candidate_contract(candidate),
        address=address,
        supporting_insertions=candidate.supporting_insertions,
    )


def _candidate_contract(candidate: Candidate) -> ContractProposal:
    return ContractProposal(symbol=candidate.symbol, prototype=candidate.prototype)


def patch_metrics(candidate: Candidate, patch: SourcePatch) -> dict[str, int]:
    return {
        "source_chars": len(candidate.source),
        "source_lines": len(candidate.source.splitlines()),
        "identifier_replacements": len(patch.identifier_replacements),
        "exact_edits": len(patch.edits),
        "old_text_chars": sum(len(edit.old) for edit in patch.edits),
        "new_text_chars": sum(len(edit.new) for edit in patch.edits),
    }

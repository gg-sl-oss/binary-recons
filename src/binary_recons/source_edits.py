"""Validation and application of the model's bounded exact source edits."""

from __future__ import annotations

import re

from .models import (
    Candidate,
    ContractProposal,
    ExactEdit,
    SourcePatch,
    SupportingInsertion,
)
from .repository import absolute_pointer_casts
from .seed import candidate_from_seed


_OPERATIONAL_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_])_*(?:DAT|FUN|PTR|UNK)_[0-9A-Fa-f]{6,16}"
    r"(?![A-Za-z0-9_])"
)
_UNRESOLVED_ADDRESS_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<token>_*(?:DAT|PTR|UNK)_(?P<address>[0-9A-Fa-f]{6,16}))"
    r"(?![A-Za-z0-9_])"
)
_DECLARED_ADDRESS_SYMBOL_RE = re.compile(
    r"\b(?P<symbol>[A-Za-z_][A-Za-z0-9_]*_"
    r"(?P<address>[0-9A-Fa-f]{6,16}))\b"
    r"\s*(?:\[\s*(?P<count>[1-9][0-9]*)\s*\])?\s*$"
)


def _unsafe_new_text(text: str) -> str | None:
    if _OPERATIONAL_IDENTIFIER_RE.search(text):
        return "reintroduces an operational decompiler identifier"
    if "Function start:" in text:
        return "changes the driver-managed function marker"
    if "```" in text or re.search(r"(?m)^\s*#", text):
        return "introduces Markdown or a preprocessor directive"
    return None


def _introduces_absolute_pointer(old: str, new: str) -> bool:
    return len(absolute_pointer_casts(new)) > len(absolute_pointer_casts(old))


def _merge_supporting_insertions(
    current: list[SupportingInsertion],
    additions: list[SupportingInsertion],
) -> list[SupportingInsertion]:
    merged = list(current)
    by_path = {insertion.path: index for index, insertion in enumerate(merged)}
    for addition in additions:
        index = by_path.get(addition.path)
        if index is None:
            by_path[addition.path] = len(merged)
            merged.append(addition)
            continue
        existing = merged[index]
        if addition.content in existing.content:
            continue
        merged[index] = SupportingInsertion(
            path=existing.path,
            content=existing.content.rstrip() + "\n\n" + addition.content,
        )
    return merged


def _declaration_element_size(prefix: str) -> int | None:
    """Return conservative Win32 sizes for simple array element declarations."""

    if "*" in prefix:
        return 4
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", prefix))
    if "__int64" in tokens or "double" in tokens:
        return 8
    if "char" in tokens or tokens & {"BYTE", "CHAR", "UCHAR"}:
        return 1
    if "short" in tokens or tokens & {"WORD", "WCHAR", "USHORT"}:
        return 2
    if tokens & {
        "int",
        "long",
        "float",
        "BOOL",
        "DWORD",
        "INT",
        "LONG",
        "UINT",
        "ULONG",
    }:
        return 4
    if any(re.fullmatch(r"H[A-Z][A-Z0-9_]*", token) for token in tokens):
        return 4
    return None


def _paired_address_objects(
    insertions: list[SupportingInsertion],
) -> list[tuple[str, int, int | None, int | None]]:
    """Read address-backed declarations present in at least two support files."""

    declarations: dict[tuple[str, int, int | None, int | None], set[str]] = {}
    for insertion in insertions:
        for raw_statement in insertion.content.split(";"):
            statement = raw_statement.strip()
            match = _DECLARED_ADDRESS_SYMBOL_RE.search(statement)
            if match is None:
                continue
            symbol = match.group("symbol")
            address = int(match.group("address"), 16)
            count_text = match.group("count")
            count = int(count_text) if count_text is not None else None
            width = (
                _declaration_element_size(statement[: match.start()])
                if count is not None
                else None
            )
            key = (symbol, address, count, width)
            declarations.setdefault(key, set()).add(insertion.path)
    return [key for key, paths in declarations.items() if len(paths) >= 2]


def bind_supporting_address_symbols(
    source: str,
    insertions: list[SupportingInsertion],
) -> tuple[str, list[str]]:
    """Propagate model-chosen paired globals to matching decompiler tokens.

    The model remains responsible for the meaningful name, type, array base,
    and array length. The driver only performs address-preserving substitutions
    that are unambiguous from the paired extern and definition.
    """

    objects = _paired_address_objects(insertions)
    changes: list[str] = []

    arrays_by_base: dict[int, set[tuple[str, int, int]]] = {}
    for symbol, address, count, width in objects:
        if count is not None and width is not None:
            arrays_by_base.setdefault(address, set()).add((symbol, count, width))
    for address, arrays in arrays_by_base.items():
        if len(arrays) != 1:
            continue
        symbol, _count, width = next(iter(arrays))
        token = (
            r"(?<![A-Za-z0-9_])_*(?:DAT|PTR|UNK)_%08X"
            r"(?![A-Za-z0-9_])" % address
        )
        pointer_index = re.compile(
            r"\*\s*\(\s*[^()\n;{}]*\*\s*\)\s*"
            r"\(\s*&\s*"
            + token
            + r"\s*\+\s*(?P<index>[A-Za-z_][A-Za-z0-9_]*)"
            + r"\s*\*\s*%d\s*\)" % width,
            re.I,
        )
        source, count = pointer_index.subn(
            lambda match: "%s[%s]" % (symbol, match.group("index")),
            source,
        )
        if count:
            changes.append(
                "address arithmetic at 0x%08X -> %s[index] (%d)"
                % (address, symbol, count)
            )

    operational = list(_UNRESOLVED_ADDRESS_TOKEN_RE.finditer(source))
    for match in operational:
        token = match.group("token")
        address = int(match.group("address"), 16)
        replacements: set[str] = set()
        for symbol, base, count, width in objects:
            if count is None:
                if address == base:
                    replacements.add(symbol)
                continue
            if address == base:
                replacements.add("%s[0]" % symbol)
                continue
            if width is None:
                continue
            offset = address - base
            if 0 < offset < count * width and offset % width == 0:
                replacements.add("%s[%d]" % (symbol, offset // width))
        if len(replacements) != 1:
            continue
        replacement = next(iter(replacements))
        pattern = re.compile(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(token))
        source, count = pattern.subn(replacement, source)
        if count:
            changes.append("%s -> %s (%d)" % (token, replacement, count))
    return source, changes


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
        if _introduces_absolute_pointer(edit.old, edit.new):
            rejected.append(
                "edit %d rejected: introduces an absolute-address pointer" % index
            )
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
        SourcePatch(
            identifier_replacements=replacements,
            edits=edits,
            supporting_insertions=patch.supporting_insertions,
        ),
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
    supporting_insertions = _merge_supporting_insertions(
        candidate.supporting_insertions,
        patch.supporting_insertions,
    )
    source, _binding_changes = bind_supporting_address_symbols(
        source,
        supporting_insertions,
    )
    return candidate_from_seed(
        source,
        contract=_candidate_contract(candidate),
        address=address,
        supporting_insertions=supporting_insertions,
    )


def _candidate_contract(candidate: Candidate) -> ContractProposal:
    return ContractProposal(symbol=candidate.symbol, prototype=candidate.prototype)


def patch_metrics(candidate: Candidate, patch: SourcePatch) -> dict[str, int]:
    return {
        "source_chars": len(candidate.source),
        "source_lines": len(candidate.source.splitlines()),
        "identifier_replacements": len(patch.identifier_replacements),
        "exact_edits": len(patch.edits),
        "supporting_insertions": len(patch.supporting_insertions),
        "supporting_chars": sum(
            len(insertion.content) for insertion in patch.supporting_insertions
        ),
        "old_text_chars": sum(len(edit.old) for edit in patch.edits),
        "new_text_chars": sum(len(edit.new) for edit in patch.edits),
    }

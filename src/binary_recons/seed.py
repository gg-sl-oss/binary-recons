"""Deterministic conversion of a Ghidra C hint into a compilable candidate seed."""

from __future__ import annotations

import re

from .models import Candidate, ContractProposal, SupportingInsertion
from .repository import ProjectRepository, normalize_candidate_marker
from .utils import read_text


_C_KEYWORDS = {
    "auto",
    "char",
    "const",
    "double",
    "enum",
    "float",
    "int",
    "long",
    "register",
    "short",
    "signed",
    "static",
    "struct",
    "union",
    "unsigned",
    "void",
    "volatile",
}

_GHIDRA_TYPE_REPLACEMENTS = {
    "undefined": "unsigned char",
    "undefined1": "unsigned char",
    "undefined2": "unsigned short",
    "undefined4": "unsigned int",
    "uint": "unsigned int",
}

_C89_REPLACEMENTS = {
    "_Bool": "int",
    "bool": "int",
    "false": "0",
    "true": "1",
    "int8_t": "char",
    "uint8_t": "unsigned char",
    "int16_t": "short",
    "uint16_t": "unsigned short",
    "int32_t": "long",
    "uint32_t": "unsigned long",
}


def _replace_token(text: str, old: str, new: str) -> tuple[str, int]:
    return re.subn(r"\b%s\b" % re.escape(old), new, text)


def _declaration_text(repository: ProjectRepository) -> str:
    return "\n".join(
        read_text(path) for path in repository.config.declarations(repository.root)
    )


def _is_declared_type(declarations: str, identifier: str) -> bool:
    typedef = re.search(
        r"\btypedef\b[^;{}]*(?:\{.*?\}\s*)?\b%s\s*;" % re.escape(identifier),
        declarations,
        re.S,
    )
    macro = re.search(
        r"(?m)^\s*#\s*define\s+%s\b" % re.escape(identifier), declarations
    )
    return typedef is not None or macro is not None


def _spelling_replacements(
    repository: ProjectRepository,
) -> dict[str, str]:
    replacements = dict(_GHIDRA_TYPE_REPLACEMENTS)
    if (
        repository.config.language.strip().lower() == "c"
        and "c89" in repository.config.rule_profiles
    ):
        replacements.update(_C89_REPLACEMENTS)
    if "Microsoft Visual C++ 4" in repository.config.compiler:
        replacements["undefined8"] = "unsigned __int64"
        replacements["longlong"] = "__int64"
        replacements["ulonglong"] = "unsigned __int64"
    else:
        replacements["undefined8"] = "unsigned long long"

    declarations = _declaration_text(repository)
    return {
        old: new
        for old, new in replacements.items()
        if not _is_declared_type(declarations, old)
    }


def normalize_contract(
    contract: ContractProposal,
    repository: ProjectRepository,
) -> ContractProposal:
    """Normalize only toolchain-incompatible decompiler spellings in a contract."""

    prototype = contract.prototype
    for old, new in _spelling_replacements(repository).items():
        prototype, _ = _replace_token(prototype, old, new)
    prototype, _ = _normalize_struct_tags(
        prototype,
        _declaration_text(repository),
    )
    return ContractProposal(symbol=contract.symbol, prototype=prototype)


def _split_parameters(signature: str) -> list[str]:
    start = signature.find("(")
    end = signature.rfind(")")
    if start < 0 or end <= start:
        return []
    parameters: list[str] = []
    beginning = start + 1
    depth = 0
    for index in range(beginning, end):
        character = signature[index]
        if character in "([":
            depth += 1
        elif character in ")]":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            parameters.append(signature[beginning:index].strip())
            beginning = index + 1
    parameters.append(signature[beginning:end].strip())
    return parameters


def _parameter_name(parameter: str) -> str | None:
    if not parameter or parameter == "void" or parameter == "...":
        return None
    function_pointer = re.search(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)", parameter)
    if function_pointer is not None:
        return function_pointer.group(1)
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", parameter)
    for identifier in reversed(identifiers):
        if identifier not in _C_KEYWORDS:
            return identifier
    return None


def _parameter_names(signature: str) -> list[str] | None:
    parameters = _split_parameters(signature)
    if len(parameters) == 1 and parameters[0] in ("", "void"):
        return []
    names = [_parameter_name(parameter) for parameter in parameters]
    if any(name is None for name in names):
        return None
    return [name for name in names if name is not None]


def _replace_identifiers(text: str, replacements: list[tuple[str, str]]) -> str:
    """Apply renames simultaneously so swapped parameter names remain safe."""

    placeholders: list[tuple[str, str]] = []
    for index, (old, new) in enumerate(replacements):
        placeholder = "__binary_recons_parameter_%d__" % index
        text = re.sub(r"\b%s\b" % re.escape(old), placeholder, text)
        placeholders.append((placeholder, new))
    for placeholder, new in placeholders:
        text = text.replace(placeholder, new)
    return text


def declaration_address_symbols(
    repository: ProjectRepository,
    excluded_address: int | None = None,
) -> dict[int, str]:
    """Return only unambiguous address-backed identifiers from project headers."""

    names: dict[int, set[str]] = {}
    for path in repository.config.declarations(repository.root):
        for line in read_text(path).splitlines():
            for name, raw_address in re.findall(
                r"\b([A-Za-z_][A-Za-z0-9_]*)_([0-9A-Fa-f]{6,8})\b",
                line,
            ):
                names.setdefault(int(raw_address, 16), set()).add(
                    name + "_" + raw_address
                )
            address_match = re.search(r"(?i)0x([0-9a-f]{6,8})", line)
            function_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if address_match is not None and function_match is not None:
                names.setdefault(int(address_match.group(1), 16), set()).add(
                    function_match.group(1)
                )
    return {
        address: next(iter(candidates))
        for address, candidates in names.items()
        if len(candidates) == 1 and address != excluded_address
    }


def _normalize_struct_tags(source: str, declarations: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    for match in re.finditer(
        r"typedef\s+struct\s+([A-Za-z_]\w*)\s*\{.*?\}\s*([A-Za-z_]\w*)\s*;",
        declarations,
        re.S,
    ):
        tag, alias = match.groups()
        if tag == alias:
            continue
        pattern = r"(?<!struct )\b%s\b" % re.escape(tag)
        source, count = re.subn(pattern, alias, source)
        if count:
            changes.append("struct tag %s -> %s (%d)" % (tag, alias, count))
    return source, changes


def normalize_decompiler_seed(
    source: str,
    repository: ProjectRepository,
    contract: ContractProposal,
    excluded_address: int | None = None,
) -> tuple[str, list[str]]:
    """Mechanically normalize a raw Ghidra definition without guessing behavior."""

    changes: list[str] = []
    brace = source.find("{")
    seed_parameters = _parameter_names(source[:brace] if brace >= 0 else source)
    contract_parameters = _parameter_names(contract.prototype)
    if (
        seed_parameters is not None
        and contract_parameters is not None
        and len(seed_parameters) == len(contract_parameters)
    ):
        renames = [
            (old, new)
            for old, new in zip(seed_parameters, contract_parameters, strict=True)
            if old != new
        ]
        if renames:
            source = _replace_identifiers(source, renames)
            changes.extend(
                "parameter %s -> %s" % replacement for replacement in renames
            )

    for old, new in _spelling_replacements(repository).items():
        source, count = _replace_token(source, old, new)
        if count:
            changes.append("source spelling %s -> %s (%d)" % (old, new, count))

    declarations = _declaration_text(repository)
    source, tag_changes = _normalize_struct_tags(source, declarations)
    changes.extend(tag_changes)

    if excluded_address is not None:
        target_token = "FUN_%08X" % excluded_address
        source, count = _replace_token(source, target_token, contract.symbol)
        if count:
            changes.append(
                "%s -> inferred target symbol %s (%d)"
                % (target_token, contract.symbol, count)
            )

    address_symbols = declaration_address_symbols(repository, excluded_address)
    operational = list(
        dict.fromkeys(re.findall(r"\b(?:DAT|FUN)_[0-9A-Fa-f]{8}\b", source))
    )
    for token in operational:
        address = int(token.rsplit("_", 1)[1], 16)
        replacement = address_symbols.get(address)
        if replacement is None or replacement == token:
            continue
        address_count = 0
        if token.startswith("DAT_"):
            source, address_count = re.subn(
                r"&\s*\b%s\b" % re.escape(token),
                "(char *)&" + replacement,
                source,
            )
        source, value_count = _replace_token(source, token, replacement)
        if address_count or value_count:
            changes.append(
                "%s -> %s (%d address, %d value)"
                % (token, replacement, address_count, value_count)
            )

    unresolved = list(dict.fromkeys(re.findall(r"\bDAT_([0-9A-Fa-f]{8})\b", source)))
    for raw_address in unresolved:
        token = "DAT_" + raw_address
        absolute = "0x" + raw_address
        source, address_count = re.subn(
            r"&\s*\b%s\b" % re.escape(token),
            "(char *)" + absolute,
            source,
        )
        source, value_count = _replace_token(
            source,
            token,
            "(*(int *)%s)" % absolute,
        )
        if address_count or value_count:
            changes.append(
                "%s -> absolute fallback (%d address, %d value)"
                % (token, address_count, value_count)
            )
    return source, changes


def candidate_from_seed(
    source: str,
    contract: ContractProposal,
    address: int,
    supporting_insertions: list[SupportingInsertion] | None = None,
) -> Candidate:
    """Lock the inferred contract onto the normalized decompiler body."""

    brace = source.find("{")
    if brace < 0:
        raise RuntimeError("decompiler hint does not contain a function body")
    candidate = Candidate(
        symbol=contract.symbol,
        prototype=contract.prototype,
        source=contract.prototype + "\n" + source[brace:],
        supporting_insertions=supporting_insertions or [],
    )
    return normalize_candidate_marker(candidate, address)


def normalize_resumed_candidate(
    candidate: Candidate,
    repository: ProjectRepository,
    address: int,
) -> tuple[Candidate, list[str]]:
    """Apply the same safe normalizations to a logged candidate before resuming."""

    contract = normalize_contract(
        ContractProposal(symbol=candidate.symbol, prototype=candidate.prototype),
        repository,
    )
    source, changes = normalize_decompiler_seed(
        candidate.source,
        repository,
        contract,
        excluded_address=address,
    )
    return (
        candidate_from_seed(
            source,
            contract,
            address,
            candidate.supporting_insertions,
        ),
        changes,
    )

"""Read project evidence and score temporary source hypotheses."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .models import (
    MAX_CANDIDATE_CHARS,
    MAX_SUPPORTING_TOTAL_CHARS,
    Candidate,
    EvidenceBundle,
    TargetSpec,
)
from .project_config import ProjectConfig, load_project_config
from .utils import atomic_write, read_text


MARKER_RE = re.compile(r"/\*\s*Function start:\s*0x([0-9A-Fa-f]+)\s*\*/")
SIMILARITY_RE = re.compile(r"Similarity:\s*([0-9]+(?:\.[0-9]+)?)%")
COMPILER_ERROR_RE = re.compile(
    r"(?:\bfatal error(?:\s+[A-Z]+\d+)?\b|"
    r"\berror\s+(?:C|LNK|U)\d{4}\b|"
    r"\bundefined reference\b|"
    r"\bunresolved external\b|"
    r":\s*error:)",
    re.I | re.M,
)
SOURCE_SAFETY_PREFIX = "SOURCE SAFETY ERROR:"
_ABSOLUTE_POINTER_RE = re.compile(
    r"(?:"
    r"\(\s*[^()\n;{}]*\*[^()\n;{}]*\)\s*(?:\(\s*)*"
    r"|(?:reinterpret_cast|static_cast|const_cast)\s*<[^>\n]*\*[^>\n]*>\s*\(\s*"
    r"|\(\s*(?:LPC?(?:VOID|STR|WSTR|BYTE|WORD|DWORD|RECT|POINT|MSG)|"
    r"PC?(?:VOID|STR|WSTR|BYTE|WORD|DWORD|RECT|POINT|MSG)|HANDLE|"
    r"H(?:WND|INSTANCE|MODULE|GLOBAL|LOCAL|BITMAP|BRUSH|CURSOR|DC|FONT|"
    r"GDIOBJ|HOOK|ICON|KEY|MENU|PALETTE|PEN|RGN|RSRC))\s*\)\s*(?:\(\s*)*"
    r"|\*\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:\(\s*)*"
    r")"
    r"(?P<address>0[xX][0-9A-Fa-f]{6,16}|[1-9][0-9]{5,})",
)
_UNRESOLVED_GLOBAL_RE = re.compile(r"\b(?:DAT|PTR|UNK)_[0-9A-Fa-f]{6,16}\b")
_C_NON_CODE_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',
    re.S,
)
SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})
ASSEMBLY_EXPORT_RE = re.compile(
    r"^FUN_([0-9A-Fa-f]+)\.disassembled\.txt$",
)
GENERIC_FUNCTION_SYMBOL_RE = re.compile(
    r"(?i)^(?:(?:FUN|sub|function|fn)_?[0-9a-f]*|helper|callback|handler|"
    r"routine|procedure|proc|dialogproc|wndproc|windowproc|eventhandler)$"
)


def is_generic_function_symbol(symbol: str) -> bool:
    return GENERIC_FUNCTION_SYMBOL_RE.fullmatch(symbol) is not None


def _source_safety_regions(candidate: Candidate) -> list[tuple[str, str]]:
    regions = [("target source", candidate.source)]
    regions.extend(
        ("support insertion %s" % insertion.path, insertion.content)
        for insertion in candidate.supporting_insertions
    )
    return regions


def _mask_c_non_code(text: str) -> str:
    """Preserve offsets and line numbers while hiding comments and literals."""

    return _C_NON_CODE_RE.sub(
        lambda match: "".join(
            "\n" if character == "\n" else " " for character in match.group(0)
        ),
        text,
    )


def absolute_pointer_casts(text: str) -> list[tuple[str, int]]:
    code = _mask_c_non_code(text)
    return [
        (match.group("address"), match.start())
        for match in _ABSOLUTE_POINTER_RE.finditer(code)
    ]


def source_text_safety_violations(text: str, region: str) -> list[str]:
    violations: list[str] = []
    code = _mask_c_non_code(text)
    for address, start in absolute_pointer_casts(text):
        line = text.count("\n", 0, start) + 1
        violations.append(
            "%s line %d uses absolute address %s as a pointer; replace it "
            "with a declared source-level global, aggregate element, or field"
            % (region, line, address)
        )
    for match in _UNRESOLVED_GLOBAL_RE.finditer(code):
        line = text.count("\n", 0, match.start()) + 1
        violations.append(
            "%s line %d still uses unresolved decompiler global %s; replace "
            "it with a meaningful declared source-level global, aggregate "
            "element, or field" % (region, line, match.group(0))
        )
    return list(dict.fromkeys(violations))


def source_safety_violations(candidate: Candidate) -> list[str]:
    """Reject decompiler address shortcuts before they can become source output."""

    violations: list[str] = []
    for region, text in _source_safety_regions(candidate):
        violations.extend(source_text_safety_violations(text, region))
    return list(dict.fromkeys(violations))


def source_safety_feedback(candidate: Candidate) -> str | None:
    violations = source_safety_violations(candidate)
    if not violations:
        return None
    return "\n".join(
        "%s %s" % (SOURCE_SAFETY_PREFIX, violation) for violation in violations[:20]
    )


def declaration_for_symbol(text: str, symbol: str) -> str | None:
    match = re.search(r"\b%s\s*\(" % re.escape(symbol), text)
    if match is None:
        return None
    start = text.rfind(";", 0, match.start()) + 1
    end = text.find(";", match.end())
    if end < 0:
        return None
    declaration = text[start:end]
    declaration = re.sub(r"/\*.*?\*/", " ", declaration, flags=re.S)
    declaration = re.sub(r"//[^\n]*", " ", declaration)
    declaration = "\n".join(
        line for line in declaration.splitlines() if not line.lstrip().startswith("#")
    )
    declaration = re.sub(r"\s+", " ", declaration).strip()
    return declaration or None


def definition_signature(text: str, symbol: str) -> str | None:
    match = re.search(
        r"^.*\b%s\s*\([^;{}]*\)\s*\{" % re.escape(symbol), text, re.M | re.S
    )
    if match is None:
        return None
    signature = match.group(0).rsplit("{", 1)[0]
    signature = re.sub(r"/\*.*?\*/", " ", signature, flags=re.S)
    signature = re.sub(r"//[^\n]*", " ", signature)
    return re.sub(r"\s+", " ", signature).strip() or None


def definition_symbol(text: str) -> str | None:
    match = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{",
        text,
        re.S,
    )
    return match.group(1) if match is not None else None


def function_span(source: str, address: int) -> tuple[int, int] | None:
    markers = list(MARKER_RE.finditer(source))
    for index, marker in enumerate(markers):
        if int(marker.group(1), 16) != address:
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(source)
        return marker.start(), end
    return None


def current_function(source: str, address: int) -> str | None:
    span = function_span(source, address)
    if span is None:
        return None
    start, end = span
    return source[start:end].rstrip()


def replace_or_insert_function(source: str, address: int, candidate: str) -> str:
    span = function_span(source, address)
    if span is not None:
        start, end = span
        tail = source[end:].lstrip("\n")
        updated = source[:start].rstrip("\n") + "\n\n" + candidate.rstrip()
        return updated + ("\n\n" + tail if tail else "\n")

    for marker in MARKER_RE.finditer(source):
        if int(marker.group(1), 16) > address:
            position = marker.start()
            return (
                source[:position].rstrip("\n")
                + "\n\n"
                + candidate.rstrip()
                + "\n\n"
                + source[position:]
            )
    return source.rstrip() + "\n\n" + candidate.rstrip() + "\n"


def normalize_candidate_marker(candidate: Candidate, address: int) -> Candidate:
    """Add the purely mechanical address marker when the model omitted it."""

    if "Function start:" in candidate.source:
        return candidate
    source = "/* Function start: 0x%X */\n%s" % (address, candidate.source)
    return candidate.model_copy(update={"source": source})


def rename_candidate_symbol(candidate: Candidate, symbol: str) -> Candidate:
    """Apply a model-proposed identifier to only the candidate's own contract."""

    pattern = r"\b%s\b" % re.escape(candidate.symbol)
    prototype, prototype_replacements = re.subn(pattern, symbol, candidate.prototype)
    source, source_replacements = re.subn(pattern, symbol, candidate.source)
    if prototype_replacements == 0 or source_replacements == 0:
        raise ValueError("candidate contract does not contain its current symbol")
    return Candidate(
        symbol=symbol,
        prototype=prototype,
        source=source,
        supporting_insertions=candidate.supporting_insertions,
    )


def validate_candidate(
    candidate: Candidate,
    target: TargetSpec,
    reserved_symbols: set[str] | None = None,
    allowed_support_paths: set[str] | None = None,
) -> None:
    source = candidate.source
    canonical_marker = "/* Function start: 0x%X */" % target.address
    errors: list[str] = []
    if len(source) > MAX_CANDIDATE_CHARS:
        errors.append(
            "candidate is over the %d-character safety limit" % MAX_CANDIDATE_CHARS
        )
    if source.count("Function start:") != 1:
        errors.append("candidate must have exactly one Function start marker")
    if canonical_marker not in source:
        errors.append("marker must be exactly %s" % canonical_marker)
    if target.symbol != candidate.symbol or target.prototype != candidate.prototype:
        errors.append("candidate fields do not match the active target contract")
    if not re.search(r"\b%s\s*\(" % re.escape(candidate.symbol), source):
        errors.append("candidate must define the target symbol")
    if is_generic_function_symbol(candidate.symbol):
        errors.append("candidate symbol must be a meaningful source-level name")
    if "%X" % target.address in candidate.symbol.upper():
        errors.append("candidate symbol must not contain the target address")
    if reserved_symbols is not None and candidate.symbol in reserved_symbols:
        errors.append("candidate symbol is already used by another function")

    compact = re.sub(r"\s+", " ", source)
    if re.sub(r"\s+", " ", candidate.prototype) not in compact:
        errors.append("signature must be exactly: %s" % candidate.prototype)

    forbidden = {
        "#include": "an include directive",
        "#define": "a preprocessor definition",
        "extern ": "an extern declaration",
        "```": "a Markdown code fence",
    }
    for token, description in forbidden.items():
        if token in source:
            errors.append("candidate contains %s" % description)
    if source.count("{") != source.count("}"):
        errors.append("candidate braces are unbalanced")

    insertion_paths: set[str] = set()
    total_insertion_chars = sum(
        len(insertion.content) for insertion in candidate.supporting_insertions
    )
    if total_insertion_chars > MAX_SUPPORTING_TOTAL_CHARS:
        errors.append(
            "supporting insertions exceed the %d-character safety limit"
            % MAX_SUPPORTING_TOTAL_CHARS
        )
    for insertion in candidate.supporting_insertions:
        if insertion.path in insertion_paths:
            errors.append("support file %s occurs more than once" % insertion.path)
        insertion_paths.add(insertion.path)
        if allowed_support_paths is None or insertion.path not in allowed_support_paths:
            errors.append("support file is not configured: %s" % insertion.path)
        if "```" in insertion.content:
            errors.append("support insertion contains a Markdown code fence")
        if "Function start:" in insertion.content:
            errors.append("support insertion contains a function marker")
        if re.search(r"(?m)^\s*#", insertion.content):
            errors.append("support insertion contains a preprocessor directive")
        if insertion.content.count("{") != insertion.content.count("}"):
            errors.append("support insertion braces are unbalanced")
        if re.search(r"\)\s*(?:const\s*)?\{", insertion.content):
            errors.append("support insertion contains a function definition")
        if re.search(r"\b%s\s*\(" % re.escape(candidate.symbol), insertion.content):
            errors.append("target prototype belongs in the managed prototype file")
        if reserved_symbols is not None:
            inserted_functions = set(
                re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", insertion.content)
            )
            for symbol in sorted(inserted_functions & reserved_symbols):
                errors.append(
                    "support insertion redeclares existing function: %s" % symbol
                )
    if errors:
        raise ValueError("; ".join(errors))


class ProjectRepository:
    def __init__(self, root: Path, config_path: Path | None = None):
        self.root = root.resolve()
        self.config: ProjectConfig = load_project_config(self.root, config_path)

    def next_unreconstructed_address(self) -> int:
        """Return the first unsafe reconstruction or unimplemented safe export."""

        if not self.config.source_units:
            raise RuntimeError(
                "cannot select the next function safely: configure source_units "
                "address ranges that exclude CRT and library code"
            )

        reconstructed: set[int] = set()
        unsafe_reconstructions: set[int] = set()
        for directory in self.config.source_paths(self.root):
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                source = read_text(path)
                for marker in MARKER_RE.finditer(source):
                    address = int(marker.group(1), 16)
                    reconstructed.add(address)
                    function = current_function(source, address)
                    if function is not None and source_text_safety_violations(
                        function,
                        "target source",
                    ):
                        unsafe_reconstructions.add(address)

        exports = self.config.resolve(self.root, self.config.exports_dir)
        if not exports.is_dir():
            raise RuntimeError("missing exports directory: %s" % exports)

        candidates: set[int] = set()
        complete_exports: set[int] = set()
        for assembly_path in exports.glob("FUN_*.disassembled.txt"):
            match = ASSEMBLY_EXPORT_RE.fullmatch(assembly_path.name)
            if match is None:
                continue
            address = int(match.group(1), 16)
            if not any(
                unit.start <= address <= unit.end for unit in self.config.source_units
            ):
                continue
            decompiled_path = exports / ("FUN_%08X.decompiled.txt" % address)
            if not decompiled_path.is_file():
                continue
            complete_exports.add(address)
            if address not in reconstructed:
                candidates.add(address)

        unsafe_candidates = unsafe_reconstructions & complete_exports
        if unsafe_candidates:
            return min(unsafe_candidates)

        if not candidates:
            raise RuntimeError(
                "no source-unsafe reconstructions or unreconstructed function "
                "exports remain inside the configured source_units ranges"
            )
        return min(candidates)

    def resolve_target(
        self,
        address: int,
        symbol: str | None = None,
        source: Path | None = None,
        prototype: str | None = None,
    ) -> TargetSpec:
        stem = "FUN_%08X" % address
        exports = self.config.resolve(self.root, self.config.exports_dir)
        assembly_path = exports / (stem + ".disassembled.txt")
        decompiled_path = exports / (stem + ".decompiled.txt")
        for path in (assembly_path, decompiled_path):
            if not path.exists():
                raise RuntimeError("missing target export: %s" % path)

        source_path = self._resolve_source(address, source)
        if not source_path.exists():
            raise RuntimeError("missing source unit: %s" % source_path)
        source_text = read_text(source_path)
        existing = current_function(source_text, address)

        if symbol is None:
            symbol = definition_symbol(existing) if existing is not None else None

        if prototype is None and symbol is not None:
            for declaration_path in self.config.declarations(self.root):
                prototype = declaration_for_symbol(read_text(declaration_path), symbol)
                if prototype is not None:
                    break
        if prototype is None and existing is not None and symbol is not None:
            prototype = definition_signature(existing, symbol)
        if (symbol is None) != (prototype is None):
            raise RuntimeError(
                "target name and prototype must be supplied together, or both left "
                "for model inference"
            )

        return TargetSpec(
            root=self.root,
            address=address,
            symbol=symbol,
            source_path=source_path,
            prototype=prototype,
            assembly_path=assembly_path,
            decompiled_path=decompiled_path,
        )

    def _resolve_source(self, address: int, source: Path | None) -> Path:
        if source is not None:
            return source if source.is_absolute() else self.root / source
        canonical_marker = "/* Function start: 0x%X */" % address
        for directory in self.config.source_paths(self.root):
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
                    if canonical_marker in read_text(path):
                        return path
        for unit in self.config.source_units:
            if unit.start <= address <= unit.end:
                return self.config.resolve(self.root, unit.path)
        raise RuntimeError(
            "could not infer a source unit for 0x%08X; add a source_units range "
            "or pass --source" % address
        )

    def collect_evidence(self, target: TargetSpec, max_callees: int) -> EvidenceBundle:
        assembly = read_text(target.assembly_path)
        decompilation = read_text(target.decompiled_path)
        return EvidenceBundle(
            language=self.config.language,
            compiler=self.config.compiler,
            project_guidance=self.config.guidance(self.root),
            original_assembly=assembly.strip(),
            decompiler_hint=self._concise_decompilation(decompilation, target.symbol),
            string_evidence=self._string_evidence(assembly, decompilation),
            reserved_symbols=self._reserved_symbol_evidence(target),
            callee_evidence=self._callee_evidence(assembly, max_callees),
            declaration_evidence=self._declaration_evidence(
                decompilation,
                excluded_address=None if target.has_contract else target.address,
            ),
            supporting_file_evidence=self._supporting_file_evidence(),
        )

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def support_file_map(self) -> dict[str, tuple[Path, str, str]]:
        result: dict[str, tuple[Path, str, str]] = {}
        for support in self.config.support_files:
            path = self.config.resolve(self.root, support.path).resolve()
            display = self._display_path(path)
            if display in result:
                raise RuntimeError("duplicate resolved support file: %s" % display)
            result[display] = (path, support.purpose, support.insertion)

        if self.config.prototype_file is not None:
            path = self.config.resolve(self.root, self.config.prototype_file).resolve()
            display = self._display_path(path)
            if display in result:
                raise RuntimeError(
                    "prototype file resolves to a configured support file: %s" % display
                )
            result[display] = (
                path,
                "Declarations for referenced non-target functions. The target "
                "declaration is managed separately and must not be inserted here.",
                "auto",
            )
        return result

    def allowed_support_paths(self) -> set[str]:
        return set(self.support_file_map())

    def _supporting_file_evidence(self, character_limit: int = 16000) -> str:
        configured = self.support_file_map()
        if not configured:
            return (
                "No support files are configured. supporting_insertions must be empty."
            )
        destinations = [
            "Allowed support destinations:",
            *(
                "- %s: %s (insertion mode: %s)" % (display, purpose, insertion)
                for display, (_, purpose, insertion) in configured.items()
            ),
        ]
        blocks: list[str] = ["\n".join(destinations)]
        prototype_path = (
            self.config.resolve(self.root, self.config.prototype_file).resolve()
            if self.config.prototype_file is not None
            else None
        )
        for display, (path, purpose, insertion) in configured.items():
            if not path.exists():
                raise RuntimeError("missing configured support file: %s" % path)
            if path == prototype_path:
                content = (
                    "<full prototype header omitted so an unimplemented target's "
                    "name and interface cannot leak; relevant declarations are "
                    "supplied separately>"
                )
            else:
                content = read_text(path)
                if len(content) > 6000:
                    content = (
                        content[:3000]
                        + "\n... mechanically truncated ...\n"
                        + content[-2800:]
                    )
            blocks.append(
                "[allowed support file: %s]\n"
                "Purpose: %s\nInsertion mode: %s\nCurrent content:\n%s"
                % (display, purpose, insertion, content.rstrip())
            )
        result = "\n\n".join(blocks)
        if len(result) <= character_limit:
            return result
        return result[:character_limit] + "\n... support-file context truncated ..."

    def _concise_decompilation(self, text: str, symbol: str | None) -> str:
        # The assembly export may carry a recovered developer name while the
        # decompiler export still carries its operational Ghidra label.  Find
        # the C definition by shape first so a name mismatch cannot discard the
        # beginning of the function.
        match = re.search(
            r"^[A-Za-z_][A-Za-z0-9_ \t:*]*\b[A-Za-z_][A-Za-z0-9_:]*"
            r"\s*\([^;{}]*?\)\s*\n\s*\{",
            text,
            re.M | re.S,
        )
        if match is not None:
            return text[match.start() :].strip()[:24000]
        if symbol is not None:
            match = re.search(r"^.*\b%s\s*\(" % re.escape(symbol), text, re.M)
            if match is not None:
                return text[match.start() :].strip()[:24000]
        return text.strip()[:24000]

    def _string_evidence(self, assembly: str, decompilation: str) -> str:
        configured = self.config.strings_file
        if configured is None:
            return "No string map is configured for this target project."
        path = self.config.resolve(self.root, configured)
        if not path.exists():
            raise RuntimeError("missing configured string map: %s" % path)

        referenced = {
            int(raw, 16)
            for raw in re.findall(
                r"(?i)(?<![0-9a-f])(?:0x|_)?([0-9a-f]{6,8})(?![0-9a-f])",
                assembly + "\n" + decompilation,
            )
        }
        evidence: list[str] = []
        for line in read_text(path).splitlines():
            match = re.match(r"(?i)^0x([0-9a-f]{6,8})\s*:", line)
            if match is not None and int(match.group(1), 16) in referenced:
                evidence.append(line.strip())
        if not evidence:
            return "No string-map entries matched this function's evidence."
        return "\n".join(evidence)[:6000]

    def reserved_symbols(self, target: TargetSpec) -> list[str]:
        symbols: set[str] = set()
        target_address = target.address
        for path in self.config.declarations(self.root):
            for line in read_text(path).splitlines():
                address_match = re.search(r"(?i)0x([0-9a-f]{6,8})", line)
                if (
                    address_match is not None
                    and int(address_match.group(1), 16) == target_address
                ):
                    continue
                match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
                if match is not None:
                    symbols.add(match.group(1))

        for directory in self.config.source_paths(self.root):
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                text = read_text(path)
                for marker in MARKER_RE.finditer(text):
                    if int(marker.group(1), 16) == target_address:
                        continue
                    tail = text[marker.end() :]
                    symbol = definition_symbol(tail)
                    if symbol is not None:
                        symbols.add(symbol)
        return sorted(symbols)

    def _reserved_symbol_evidence(self, target: TargetSpec) -> str:
        symbols = self.reserved_symbols(target)
        if not symbols:
            return "No source-level function names are reserved yet."
        return (
            "Do not reuse any of these names; their interfaces are intentionally "
            "not supplied:\n" + ", ".join(symbols[:200])
        )

    def _callee_evidence(self, assembly: str, max_callees: int) -> str:
        addresses: list[int] = []
        for raw in re.findall(r"\bCALL\s+0x([0-9A-Fa-f]+)", assembly, re.I):
            address = int(raw, 16)
            if address not in addresses:
                addresses.append(address)
        headers = [read_text(path) for path in self.config.declarations(self.root)]
        evidence: list[str] = []
        exports = self.config.resolve(self.root, self.config.exports_dir)
        for address in addresses[:max_callees]:
            path = exports / ("FUN_%08X.disassembled.txt" % address)
            if not path.exists():
                evidence.append("CALL 0x%08X: no developer export available" % address)
                continue
            callee = read_text(path).strip()
            if len(callee) > 2400:
                callee = (
                    callee[:1800] + "\n... mechanically truncated ...\n" + callee[-500:]
                )
            name_match = re.search(r"^Function:\s*(\w+)", callee, re.M)
            declaration = None
            if name_match is not None:
                for header in headers:
                    declaration = declaration_for_symbol(header, name_match.group(1))
                    if declaration is not None:
                        break
            block = [callee]
            if declaration is not None:
                block.append("Central declaration: %s;" % declaration)
            evidence.append("\n".join(block))
        return (
            "\n\n".join(evidence)
            if evidence
            else "No direct CALL targets were present."
        )

    def _declaration_evidence(
        self,
        decompilation: str,
        excluded_address: int | None = None,
    ) -> str:
        paths = self.config.declarations(self.root)
        if not paths:
            return "No declaration files are configured for this target project."

        ignored = {
            "auto",
            "break",
            "case",
            "__cdecl",
            "__fastcall",
            "__stdcall",
            "__thiscall",
            "char",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "goto",
            "if",
            "int",
            "long",
            "register",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "union",
            "unsigned",
            "undefined",
            "undefined1",
            "undefined2",
            "undefined4",
            "undefined8",
            "void",
            "volatile",
            "while",
        }
        identifiers = (
            set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", decompilation)) - ignored
        )
        fields = set(re.findall(r"(?:\.|->)([A-Za-z_][A-Za-z0-9_]*)", decompilation))
        address_tokens = {
            match.group(1).lower()
            for identifier in identifiers
            if (match := re.search(r"_([0-9A-Fa-f]{6,8})$", identifier))
        }

        evidence: list[str] = []
        matched_lines: list[str] = []
        for path in paths:
            for line_number, line in enumerate(read_text(path).splitlines(), 1):
                line_identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", line))
                direct_matches = identifiers & line_identifiers
                line_addresses = {
                    token.lower()
                    for token in re.findall(
                        r"(?i)(?<![0-9a-f])([0-9a-f]{6,8})(?![0-9a-f])", line
                    )
                }
                if excluded_address is not None and excluded_address in {
                    int(token, 16) for token in line_addresses
                }:
                    continue
                if not direct_matches and not (address_tokens & line_addresses):
                    continue
                try:
                    display = path.relative_to(self.root)
                except ValueError:
                    display = path
                stripped = line.strip()
                evidence.append("%s:%d: %s" % (display, line_number, stripped))
                matched_lines.append(stripped)

        declaration_text = "\n".join(matched_lines)
        for path in paths:
            header = read_text(path)
            for match in re.finditer(
                r"typedef\s+struct(?:\s+[A-Za-z_][A-Za-z0-9_]*)?\s*\{"
                r".*?\}\s*([A-Za-z_][A-Za-z0-9_]*)\s*;",
                header,
                re.S,
            ):
                type_name = match.group(1)
                if re.search(r"\b%s\b" % re.escape(type_name), declaration_text):
                    try:
                        display = path.relative_to(self.root)
                    except ValueError:
                        display = path
                    evidence.append(
                        "%s relevant type definition:\n%s"
                        % (display, match.group(0).strip())
                    )

        unresolved = sorted(
            identifier
            for identifier in identifiers
            if re.search(r"_[0-9A-Fa-f]{6,8}$", identifier)
            and not any(
                re.search(r"\b%s\b" % re.escape(identifier), line)
                for line in matched_lines
            )
        )
        if unresolved:
            evidence.append(
                "Decompiler identifiers with no exact configured declaration: %s"
                % ", ".join(unresolved[:40])
            )
        if fields and not evidence:
            evidence.append("Referenced members: %s" % ", ".join(sorted(fields)))
        if not evidence:
            return "No configured declarations matched this decompilation."
        return "\n".join(evidence)[:12000]

    def compare(self, target: TargetSpec, timeout: float) -> tuple[float | None, str]:
        if target.symbol is None:
            raise RuntimeError("cannot compare a target without a proposed symbol")
        command = self.config.comparison_command(target.symbol, target.address)
        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return None, output + "\nBUILD/COMPARE TIMED OUT"
        match = SIMILARITY_RE.search(result.stdout)
        score = float(match.group(1)) if match is not None else None
        return score, result.stdout

    def snapshot_workspace(self, target: TargetSpec) -> dict[Path, str]:
        """Capture every file a candidate evaluation is permitted to change."""

        paths = {target.source_path.resolve()}
        if self.config.prototype_file is not None:
            paths.add(
                self.config.resolve(self.root, self.config.prototype_file).resolve()
            )
        paths.update(path for path, _, _ in self.support_file_map().values())
        missing = sorted(str(path) for path in paths if not path.exists())
        if missing:
            raise RuntimeError(
                "missing candidate workspace file(s): %s" % ", ".join(missing)
            )
        return {path: read_text(path) for path in sorted(paths)}

    def render_candidate_workspace(
        self,
        target: TargetSpec,
        candidate: Candidate,
        baseline: dict[Path, str],
    ) -> dict[Path, str]:
        """Render a complete candidate change set without touching the filesystem."""

        if not target.has_contract:
            raise RuntimeError("cannot render a candidate without an active contract")
        workspace = dict(baseline)
        source_path = target.source_path.resolve()
        if source_path not in workspace:
            raise RuntimeError("target source is outside the candidate workspace")
        workspace[source_path] = replace_or_insert_function(
            workspace[source_path], target.address, candidate.source
        )

        if self.config.prototype_file is not None:
            prototype_path = self.config.resolve(
                self.root, self.config.prototype_file
            ).resolve()
            if prototype_path not in workspace:
                raise RuntimeError("prototype file is outside the candidate workspace")
            workspace[prototype_path] = self._render_contract(
                workspace[prototype_path], target
            )

        configured = self.support_file_map()
        for insertion in candidate.supporting_insertions:
            if insertion.path not in configured:
                raise RuntimeError(
                    "candidate references an unconfigured support file: %s"
                    % insertion.path
                )
            support_path, _, mode = configured[insertion.path]
            if support_path == source_path:
                raise RuntimeError(
                    "target source cannot also receive a supporting insertion"
                )
            workspace[support_path] = self._insert_supporting_content(
                workspace[support_path], insertion.content, mode, support_path
            )
        return workspace

    @staticmethod
    def apply_workspace(workspace: dict[Path, str]) -> None:
        for path in sorted(workspace):
            if path.exists() and read_text(path) == workspace[path]:
                continue
            atomic_write(path, workspace[path])

    def changed_workspace_files(
        self,
        baseline: dict[Path, str],
        candidate: dict[Path, str],
    ) -> list[str]:
        return [
            self._display_path(path)
            for path in sorted(candidate)
            if candidate[path] != baseline[path]
        ]

    @staticmethod
    def _insert_supporting_content(
        source: str,
        insertion: str,
        mode: str,
        path: Path,
    ) -> str:
        snippet = insertion.strip()
        if snippet in source:
            return source
        if mode == "auto":
            mode = (
                "before-final-endif"
                if path.suffix.lower() in {".h", ".hh", ".hpp", ".hxx"}
                and re.search(r"(?m)^\s*#\s*endif\b", source)
                else "append"
            )
        if mode == "append":
            return source.rstrip() + "\n\n" + snippet + "\n"
        if mode == "before-final-endif":
            matches = list(re.finditer(r"(?m)^\s*#\s*endif\b", source))
            if not matches:
                raise RuntimeError(
                    "support file has no final #endif insertion point: %s" % path
                )
            position = matches[-1].start()
            return (
                source[:position].rstrip()
                + "\n\n"
                + snippet
                + "\n\n"
                + source[position:].lstrip("\n")
            )
        raise RuntimeError("unknown support insertion mode: %s" % mode)

    @staticmethod
    def _render_contract(source: str, target: TargetSpec) -> str:
        if not target.has_contract:
            return source
        assert target.symbol is not None
        assert target.prototype is not None
        lines = source.splitlines()
        declaration = "%s; /* 0x%08X */" % (target.prototype, target.address)
        address_pattern = re.compile(r"(?i)0x0*%x\b" % target.address)
        for index, line in enumerate(lines):
            if address_pattern.search(line):
                lines[index] = declaration
                return "\n".join(lines) + "\n"
        if any(
            re.search(r"\b%s\s*\(" % re.escape(target.symbol), line) for line in lines
        ):
            return source

        insertion = next(
            (
                index
                for index, line in enumerate(lines)
                if (match := re.search(r"(?i)0x([0-9a-f]{6,8})", line))
                and int(match.group(1), 16) > target.address
            ),
            next(
                (index for index, line in enumerate(lines) if line.strip() == "#endif"),
                len(lines),
            ),
        )
        lines.insert(insertion, declaration)
        return "\n".join(lines) + "\n"

    def persist_contract(self, target: TargetSpec) -> None:
        """Write a winning model-proposed prototype to the configured header."""

        if not target.has_contract or self.config.prototype_file is None:
            return
        assert target.symbol is not None
        assert target.prototype is not None
        path = self.config.resolve(self.root, self.config.prototype_file)
        if not path.exists():
            raise RuntimeError("missing configured prototype file: %s" % path)

        atomic_write(path, self._render_contract(read_text(path), target))

    @staticmethod
    def compact_feedback(output: str, score: float | None) -> str:
        lines = output.strip().splitlines()
        if score is not None:
            selected = [
                line
                for line in lines
                if "Comparison for function" in line
                or " | " in line
                or "Similarity:" in line
            ]
            return "\n".join(selected)[-9000:]
        safety_errors = [
            line for line in lines if line.startswith(SOURCE_SAFETY_PREFIX)
        ]
        if safety_errors:
            return "\n".join(safety_errors)[-6000:]
        errors = [line for line in lines if COMPILER_ERROR_RE.search(line)]
        if errors:
            # Warnings often precede the first fatal diagnostic in historical
            # compiler output.  A bounded compile-only turn must spend its one
            # edit on an actual blocker, not on an earlier harmless warning.
            return "\n".join(errors)[-6000:]
        selected = [
            line
            for line in lines
            if re.search(r"error|warning|make:|timed out", line, re.I)
        ]
        if not selected:
            selected = lines[-30:]
        return "\n".join(selected)[-6000:]

    @staticmethod
    def compact_compile_feedback(output: str, rendered_source: str | None) -> str:
        """Pair the first fatal diagnostic with its exact rendered source line."""

        feedback = ProjectRepository.compact_feedback(output, None)
        if rendered_source is None:
            return feedback
        diagnostic = next(
            (line for line in output.splitlines() if COMPILER_ERROR_RE.search(line)),
            None,
        )
        if diagnostic is None:
            return feedback
        line_match = re.search(r"\((\d+)\)\s*:", diagnostic)
        if line_match is None:
            line_match = re.search(r":(\d+)(?::\d+)?:\s*(?:fatal\s+)?error", diagnostic)
        if line_match is None:
            return feedback
        line_number = int(line_match.group(1))
        source_lines = rendered_source.splitlines()
        if line_number < 1 or line_number > len(source_lines):
            return feedback
        beginning = max(1, line_number - 2)
        ending = min(len(source_lines), line_number + 2)
        nearby = "\n".join(
            "%d: %s" % (number, source_lines[number - 1])
            for number in range(beginning, ending + 1)
        )
        return (
            feedback
            + "\n\nFIRST BLOCKING SOURCE FILE LINE (verbatim):\n"
            + source_lines[line_number - 1]
            + "\n\nNUMBERED NEARBY SOURCE (numbers are annotations):\n"
            + nearby
        )[-9000:]

    @staticmethod
    def compact_similarity_feedback(output: str) -> str:
        """Keep mismatched instruction rows while normalizing relocated addresses."""

        lines = output.splitlines()
        mismatches: list[str] = []
        header = next(
            (line for line in lines if "Comparison for function" in line),
            "",
        )

        def instruction_shape(value: str) -> str:
            value = value.split(":", 1)[-1]
            value = re.sub(r"0x[0-9A-Fa-f]{5,}", "0xADDR", value)
            return re.sub(r"\s+", " ", value).strip().lower()

        for line in lines:
            if " | " not in line:
                continue
            left, right = line.split(" | ", 1)
            if instruction_shape(left) != instruction_shape(right):
                mismatches.append(line)

        similarity = next(
            (line for line in reversed(lines) if "Similarity:" in line),
            "",
        )
        selected = [header] if header else []
        selected.extend(mismatches[:24])
        if len(mismatches) > 24:
            selected.extend(["... later mismatch rows omitted ...", *mismatches[-8:]])
        if similarity:
            selected.append(similarity)
        if not selected:
            return ProjectRepository.compact_feedback(output, None)
        return "\n".join(selected)[-9000:]

    @staticmethod
    def has_compiler_errors(output: str) -> bool:
        return COMPILER_ERROR_RE.search(output) is not None

    @staticmethod
    def has_source_safety_errors(output: str) -> bool:
        return SOURCE_SAFETY_PREFIX in output

    @staticmethod
    def is_repairable_build_failure(output: str) -> bool:
        """Allow repair for compiler diagnostics and hard source-safety gates."""

        return "BUILD/COMPARE TIMED OUT" not in output and (
            ProjectRepository.has_compiler_errors(output)
            or ProjectRepository.has_source_safety_errors(output)
        )

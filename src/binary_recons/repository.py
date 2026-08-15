"""Read project evidence and score temporary source hypotheses."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from .models import MAX_CANDIDATE_CHARS, Candidate, EvidenceBundle, TargetSpec
from .project_config import ProjectConfig, load_project_config
from .utils import atomic_write, read_text


MARKER_RE = re.compile(r"/\*\s*Function start:\s*0x([0-9A-Fa-f]+)\s*\*/")
SIMILARITY_RE = re.compile(r"Similarity:\s*([0-9]+(?:\.[0-9]+)?)%")
COMPILER_ERROR_RE = re.compile(
    r"(?:\bfatal error(?:\s+[A-Z]+\d+)?\b|"
    r"\berror\s+(?:C|LNK|U)\d{4}\b|"
    r":\s*error:)",
    re.I | re.M,
)
SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})


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


def candidate_fingerprint(candidate: str) -> str:
    normalized = re.sub(r"\s+", " ", candidate).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_candidate_marker(candidate: Candidate, address: int) -> Candidate:
    """Add the purely mechanical address marker when the model omitted it."""

    if "Function start:" in candidate.source:
        return candidate
    source = "/* Function start: 0x%X */\n%s" % (address, candidate.source)
    return candidate.model_copy(update={"source": source})


def validate_candidate(
    candidate: Candidate,
    target: TargetSpec,
    reserved_symbols: set[str] | None = None,
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
    if re.match(r"(?i)^(?:FUN|sub|function|fn)_?[0-9a-f]*$", candidate.symbol):
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
    if errors:
        raise ValueError("; ".join(errors))


class ProjectRepository:
    def __init__(self, root: Path, config_path: Path | None = None):
        self.root = root.resolve()
        self.config: ProjectConfig = load_project_config(self.root, config_path)

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
        )

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
            return text[match.start() :].strip()[:6000]
        if symbol is not None:
            match = re.search(r"^.*\b%s\s*\(" % re.escape(symbol), text, re.M)
            if match is not None:
                return text[match.start() :].strip()[:6000]
        return text.strip()[:6000]

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

    def persist_contract(self, target: TargetSpec) -> None:
        """Write a winning model-proposed prototype to the configured header."""

        if not target.has_contract or self.config.prototype_file is None:
            return
        assert target.symbol is not None
        assert target.prototype is not None
        path = self.config.resolve(self.root, self.config.prototype_file)
        if not path.exists():
            raise RuntimeError("missing configured prototype file: %s" % path)

        lines = read_text(path).splitlines()
        declaration = "%s; /* 0x%08X */" % (target.prototype, target.address)
        address_pattern = re.compile(r"(?i)0x0*%x\b" % target.address)
        for index, line in enumerate(lines):
            if address_pattern.search(line):
                lines[index] = declaration
                atomic_write(path, "\n".join(lines) + "\n")
                return
        if any(
            re.search(r"\b%s\s*\(" % re.escape(target.symbol), line) for line in lines
        ):
            return

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
        atomic_write(path, "\n".join(lines) + "\n")

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
        selected = [
            line
            for line in lines
            if re.search(r"error|warning|make:|timed out", line, re.I)
        ]
        if not selected:
            selected = lines[-30:]
        return "\n".join(selected)[-6000:]

    @staticmethod
    def has_compiler_errors(output: str) -> bool:
        return COMPILER_ERROR_RE.search(output) is not None

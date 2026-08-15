"""Read project evidence and score temporary source hypotheses."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from .models import MAX_CANDIDATE_CHARS, EvidenceBundle, TargetSpec
from .utils import read_text


MARKER_RE = re.compile(r"/\*\s*Function start:\s*0x([0-9A-Fa-f]+)\s*\*/")
SIMILARITY_RE = re.compile(r"Similarity:\s*([0-9]+(?:\.[0-9]+)?)%")


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
        return source[:start].rstrip("\n") + "\n\n" + candidate.rstrip() + "\n\n" + tail

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


def validate_candidate(candidate: str, target: TargetSpec) -> None:
    canonical_marker = "/* Function start: 0x%X */" % target.address
    errors: list[str] = []
    if len(candidate) > MAX_CANDIDATE_CHARS:
        errors.append(
            "candidate is over the %d-character safety limit" % MAX_CANDIDATE_CHARS
        )
    if candidate.count("Function start:") != 1:
        errors.append("candidate must have exactly one Function start marker")
    if canonical_marker not in candidate:
        errors.append("marker must be exactly %s" % canonical_marker)
    if candidate.count(target.symbol) != 1:
        errors.append("candidate must define the target symbol exactly once")

    compact = re.sub(r"\s+", " ", candidate)
    if re.sub(r"\s+", " ", target.prototype) not in compact:
        errors.append("signature must be exactly: %s" % target.prototype)

    forbidden = {
        "__asm": "inline assembly",
        "__declspec(naked)": "a naked function",
        "this->": "this->",
        "#include": "an include directive",
        "#define": "a preprocessor definition",
        "extern ": "an extern declaration",
        "```": "a Markdown code fence",
    }
    for token, description in forbidden.items():
        if token in candidate:
            errors.append("candidate contains %s" % description)
    if re.search(r"\b(?:try|catch|__finally)\b", candidate):
        errors.append("candidate contains forbidden exception handling")
    if candidate.count("{") != candidate.count("}"):
        errors.append("candidate braces are unbalanced")
    if re.search(r"\{\s*(?:return|[A-Za-z_]\w*\s*=)[^\n]*\}", candidate):
        errors.append("candidate puts the function body on one line")
    if errors:
        raise ValueError("; ".join(errors))


class ProjectRepository:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve_target(
        self,
        address: int,
        symbol: str | None = None,
        source: Path | None = None,
        prototype: str | None = None,
    ) -> TargetSpec:
        stem = "FUN_%08X" % address
        assembly_path = self.root / "code-full" / (stem + ".disassembled.txt")
        decompiled_path = self.root / "code-full" / (stem + ".decompiled.txt")
        for path in (assembly_path, decompiled_path):
            if not path.exists():
                raise RuntimeError("missing target export: %s" % path)

        assembly = read_text(assembly_path)
        if symbol is None:
            match = re.search(r"^Function:\s*([A-Za-z_]\w*)", assembly, re.M)
            if match is None:
                raise RuntimeError(
                    "could not infer target symbol from %s" % assembly_path
                )
            symbol = match.group(1)

        implemented_path = self.root / "include/wc1funcs.h"
        external_path = self.root / "include/wc1extern.h"
        implemented = read_text(implemented_path)
        external = read_text(external_path)
        if not re.search(r"\b%s\s*\(" % re.escape(symbol), implemented):
            raise RuntimeError("target is not declared in include/wc1funcs.h")
        if re.search(r"\b%s\s*\(" % re.escape(symbol), external):
            raise RuntimeError("target is still declared in include/wc1extern.h")
        if prototype is None:
            prototype = declaration_for_symbol(implemented, symbol)
        if prototype is None:
            raise RuntimeError("could not recover the target prototype")

        source_path = self._resolve_source(address, source)
        if not source_path.exists():
            raise RuntimeError("missing source unit: %s" % source_path)

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
        order = read_text(self.root / "docs/ORDER.md")
        pattern = re.compile(
            r"\|\s*`(?P<path>src/[^`]+)`\s*\|\s*"
            r"`0x(?P<start>[0-9A-Fa-f]+)`\s*[–-]\s*"
            r"`0x(?P<end>[0-9A-Fa-f]+)`"
        )
        for match in pattern.finditer(order):
            start = int(match.group("start"), 16)
            end = int(match.group("end"), 16)
            if start <= address <= end:
                return self.root / match.group("path")
        raise RuntimeError(
            "could not infer a source unit for 0x%08X from docs/ORDER.md" % address
        )

    def collect_evidence(self, target: TargetSpec, max_callees: int) -> EvidenceBundle:
        assembly = read_text(target.assembly_path)
        decompilation = read_text(target.decompiled_path)
        return EvidenceBundle(
            original_assembly=assembly.strip(),
            decompiler_hint=self._concise_decompilation(decompilation, target.symbol),
            callee_evidence=self._callee_evidence(assembly, max_callees),
            declaration_evidence=self._declaration_evidence(decompilation),
        )

    def _concise_decompilation(self, text: str, symbol: str) -> str:
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
        match = re.search(r"^.*\b%s\s*\(" % re.escape(symbol), text, re.M)
        if match is not None:
            return text[match.start() :].strip()[:6000]
        return text.strip()[:6000]

    def _callee_evidence(self, assembly: str, max_callees: int) -> str:
        addresses: list[int] = []
        for raw in re.findall(r"\bCALL\s+0x([0-9A-Fa-f]+)", assembly, re.I):
            address = int(raw, 16)
            if address not in addresses:
                addresses.append(address)
        headers = [
            read_text(self.root / "include/wc1funcs.h"),
            read_text(self.root / "include/wc1extern.h"),
        ]
        evidence: list[str] = []
        for address in addresses[:max_callees]:
            path = self.root / "code-full" / ("FUN_%08X.disassembled.txt" % address)
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

    def _declaration_evidence(self, decompilation: str) -> str:
        headers = sorted((self.root / "include").glob("*.h"))
        header_lines = [(path, read_text(path).splitlines()) for path in headers]
        declarations_by_address: dict[str, list[tuple[Path, int, str]]] = {}
        for path, lines in header_lines:
            for line_number, line in enumerate(lines, 1):
                for declared in re.findall(
                    r"\b(?:g_[A-Za-z_][A-Za-z0-9_]*|DAT_[0-9A-Fa-f]+)\b",
                    line,
                ):
                    address_match = re.search(r"_([0-9A-Fa-f]{8})$", declared)
                    if address_match is None:
                        continue
                    declarations_by_address.setdefault(
                        address_match.group(1).lower(), []
                    ).append((path, line_number, line.strip()))
        identifiers = list(
            dict.fromkeys(
                re.findall(
                    r"\b(?:g_[A-Za-z_][A-Za-z0-9_]*|DAT_[0-9A-Fa-f]+)\b",
                    decompilation,
                )
            )
        )
        fields = list(
            dict.fromkeys(
                re.findall(r"(?:\.|->)([A-Za-z_][A-Za-z0-9_]*)", decompilation)
            )
        )
        evidence: list[str] = []
        matched_declarations: list[str] = []

        for identifier in identifiers:
            found = False
            for path, lines in header_lines:
                for line_number, line in enumerate(lines, 1):
                    if re.search(r"\b%s\b" % re.escape(identifier), line):
                        matched_declarations.append(line)
                        evidence.append(
                            "%s:%d: %s"
                            % (
                                path.relative_to(self.root),
                                line_number,
                                line.strip(),
                            )
                        )
                        found = True
                        break
                if found:
                    break
            if not found:
                address_match = re.search(r"_([0-9A-Fa-f]{8})$", identifier)
                address_matches = []
                if address_match is not None:
                    address_matches = declarations_by_address.get(
                        address_match.group(1).lower(), []
                    )
                if address_matches:
                    for path, line_number, line in address_matches[:3]:
                        matched_declarations.append(line)
                        evidence.append(
                            "Address-matched central declaration for %s: %s:%d: %s"
                            % (
                                identifier,
                                path.relative_to(self.root),
                                line_number,
                                line,
                            )
                        )
                else:
                    evidence.append("No central declaration found for %s" % identifier)

        for field in fields:
            found = False
            for path, lines in header_lines:
                for line_number, line in enumerate(lines, 1):
                    if re.search(r"\b%s\b" % re.escape(field), line) and ";" in line:
                        evidence.append(
                            "%s:%d member: %s"
                            % (
                                path.relative_to(self.root),
                                line_number,
                                line.strip(),
                            )
                        )
                        found = True
                        break
                if found:
                    break

        declaration_text = "\n".join(matched_declarations)
        for path in headers:
            header = read_text(path)
            for match in re.finditer(
                r"typedef\s+struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{"
                r".*?\}\s*\1\s*;",
                header,
                re.S,
            ):
                type_name = match.group(1)
                if re.search(r"\b%s\b" % re.escape(type_name), declaration_text):
                    evidence.append(
                        "%s relevant type definition:\n%s"
                        % (path.relative_to(self.root), match.group(0).strip())
                    )

        if not evidence:
            return "No named global or member references were present."
        return "\n".join(evidence)[:12000]

    def compare(self, target: TargetSpec, timeout: float) -> tuple[float | None, str]:
        command = ["make", "compare-func", "FUNC=" + target.symbol]
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

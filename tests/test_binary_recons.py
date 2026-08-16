"""Model-free tests for the compile-first reconstruction workflow."""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from binary_recons.cli import build_parser, entrypoint, main  # noqa: E402
from binary_recons.llama_server import ManagedLlamaServer  # noqa: E402
from binary_recons.model_client import (  # noqa: E402
    ModelRequestError,
    StructuredModelClient,
)
from binary_recons.models import (  # noqa: E402
    Candidate,
    ContractProposal,
    ExactEdit,
    IdentifierReplacement,
    LlamaServerConfig,
    ModelPreset,
    SearchConfig,
    ServerMode,
    SimilarityPatch,
    SourcePatch,
    SupportingInsertion,
    SymbolProposalBatch,
)
from binary_recons.repository import (  # noqa: E402
    ProjectRepository,
    candidate_fingerprint,
    current_function,
    normalize_candidate_marker,
    rename_candidate_symbol,
    replace_or_insert_function,
    validate_candidate,
)
from binary_recons.search import (  # noqa: E402
    ReconstructionSearch,
    _select_symbol_proposal,
)
from binary_recons.seed import (  # noqa: E402
    candidate_from_seed,
    declaration_address_symbols,
    normalize_contract,
    normalize_decompiler_seed,
)
from binary_recons.source_edits import (  # noqa: E402
    apply_source_patch,
    sanitize_source_patch,
)


FAKE_SERVER = r"""
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/health', '/v1/health'):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""


def write_fixture(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_fixture_project(root: Path, decompiled_body: str = "return 7;") -> None:
    write_fixture(
        root,
        "binary-recons.toml",
        'language = "C"\n'
        'compiler = "Microsoft Visual C++ 4.20"\n'
        'exports_dir = "analysis"\n'
        'output_dir = "artifacts/reconstruction"\n'
        'strings_file = "analysis/strings.txt"\n'
        'source_dirs = ["src"]\n'
        'declaration_files = ["include/*.h"]\n'
        'prototype_file = "include/functions.h"\n'
        'rule_profiles = ["c89", "msvc4-od"]\n'
        'prompt_files = ["RECONSTRUCTION.md"]\n'
        'compare_command = ["fixture-compare", "{symbol}", "{address_hex}"]\n'
        "\n"
        "[[support_files]]\n"
        'path = "include/types.h"\n'
        'purpose = "Shared source-level type declarations."\n'
        "\n"
        "[[support_files]]\n"
        'path = "include/globals.h"\n'
        'purpose = "Extern declarations for evidenced globals."\n'
        "\n"
        "[[support_files]]\n"
        'path = "src/globals.c"\n'
        'purpose = "Definitions matching newly declared globals."\n'
        "\n"
        "[[source_units]]\n"
        'path = "src/sample.c"\n'
        "start = 0x401000\n"
        "end = 0x4010ff\n",
    )
    write_fixture(root, "RECONSTRUCTION.md", "Use portable fixture source.\n")
    write_fixture(root, "analysis/strings.txt", '0x00403000: "fixture text"\n')
    write_fixture(root, "src/sample.c", '#include "project.h"\n')
    write_fixture(
        root,
        "include/functions.h",
        "int HiddenTargetStub(void); /* 0x00401000 */\n"
        "void ExistingFixtureAction(void); /* 0x00402000 */\n",
    )
    write_fixture(
        root,
        "include/globals.h",
        "#ifndef FIXTURE_GLOBALS_H\n"
        "#define FIXTURE_GLOBALS_H\n\n"
        "extern int g_fixture_value_00402010[4];\n"
        "extern FixtureVector g_fixture_vector_00402020[4];\n\n"
        "#endif\n",
    )
    write_fixture(
        root,
        "include/types.h",
        "#ifndef FIXTURE_TYPES_H\n#define FIXTURE_TYPES_H\n\n#endif\n",
    )
    write_fixture(root, "src/globals.c", '#include "project.h"\n')
    write_fixture(
        root,
        "analysis/FUN_00401000.disassembled.txt",
        "Function: FUN_00401000\n"
        "Address: 0x00401000\n\n"
        "MOV EAX,0x7\n"
        "PUSH 0x403000\n"
        "CMP EAX,0x7\n"
        "RET\n",
    )
    write_fixture(
        root,
        "analysis/FUN_00401000.decompiled.txt",
        "Function: FUN_00401000\n"
        "Address: 0x00401000\n\n"
        "int FUN_00401000(void)\n\n"
        "{\n" + "  %s\n" % decompiled_body + "}\n",
    )


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_ready(port: int) -> None:
    import httpx

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if httpx.get("http://127.0.0.1:%d/health" % port).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.02)
    raise AssertionError("fake server did not become ready")


class FakeManagedServer:
    def __init__(self, *args: object):
        pass

    def __enter__(self) -> "FakeManagedServer":
        return self

    def __exit__(self, *args: object) -> None:
        pass


class RepositoryTests(unittest.TestCase):
    def test_model_request_failure_has_a_concise_cli_error(self) -> None:
        stderr = io.StringIO()
        with (
            patch(
                "binary_recons.cli.main",
                side_effect=ModelRequestError("Request timed out."),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            result = entrypoint()
        self.assertEqual(result, 1)
        self.assertEqual(stderr.getvalue(), "binary-recons: Request timed out.\n")

    def test_new_target_contract_is_not_taken_from_the_stub_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            self.assertIsNone(target.symbol)
            self.assertIsNone(target.prototype)
            self.assertEqual(
                repository.reserved_symbols(target), ["ExistingFixtureAction"]
            )

    def test_cli_dry_run_writes_only_the_contract_prompt_for_a_new_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            with (
                patch("binary_recons.cli.DEFAULT_MODEL_PATH", None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = main(
                    [
                        "--project-root",
                        str(root),
                        "--address",
                        "0x401000",
                        "--dry-run-prompt",
                    ]
                )
            self.assertEqual(result, 0)
            prompts = list(
                (root / "artifacts/reconstruction/local-model/00401000").glob(
                    "*/contract.prompt.txt"
                )
            )
            self.assertEqual(len(prompts), 1)
            prompt = prompts[0].read_text(encoding="utf-8")
            self.assertIn("Infer only a meaningful source-level function name", prompt)
            self.assertIn("[shared profile: c89]", prompt)
            self.assertIn("[project file: RECONSTRUCTION.md]", prompt)
            self.assertIn('0x00403000: "fixture text"', prompt)
            self.assertIn("ExistingFixtureAction", prompt)
            self.assertNotIn("HiddenTargetStub", prompt)

    def test_cli_has_no_alternate_draft_source_option(self) -> None:
        options = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--draft-source", options)
        self.assertIn("--max-edits", options)
        self.assertIn("--max-iterations", options)

    def test_existing_definition_keeps_its_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            write_fixture(
                root,
                "src/sample.c",
                """/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return 7;
}
""",
            )
            target = ProjectRepository(root).resolve_target(0x00401000)
            self.assertEqual(target.symbol, "ReadFixtureValue")
            self.assertEqual(target.prototype, "int ReadFixtureValue(void)")

    def test_reopen_contract_hides_the_existing_weak_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            write_fixture(
                root,
                "src/sample.c",
                """/* Function start: 0x401000 */
int DialogProc(void)
{
    return 7;
}
""",
            )
            with (
                patch("binary_recons.cli.DEFAULT_MODEL_PATH", None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = main(
                    [
                        "--project-root",
                        str(root),
                        "--address",
                        "0x401000",
                        "--reopen-contract",
                        "--dry-run-prompt",
                    ]
                )
            self.assertEqual(result, 0)
            prompt = next(
                (root / "artifacts/reconstruction/local-model/00401000").glob(
                    "*/contract.prompt.txt"
                )
            ).read_text(encoding="utf-8")
            self.assertNotIn("DialogProc", prompt)

    def test_bare_mechanism_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            candidate = Candidate(
                symbol="DialogProc",
                prototype="int DialogProc(void)",
                source="""/* Function start: 0x401000 */
int DialogProc(void)
{
    return 7;
}""",
            )
            with self.assertRaisesRegex(ValueError, "meaningful source-level name"):
                validate_candidate(
                    candidate,
                    target.with_candidate_contract(candidate),
                    set(repository.reserved_symbols(target)),
                    repository.allowed_support_paths(),
                )
            self.assertEqual(
                _select_symbol_proposal(
                    ["DialogProc", "MeasureFixtureState", "ReadFixtureState"],
                    set(),
                    0x00401000,
                ),
                "MeasureFixtureState",
            )

    def test_function_insertion_is_address_sorted(self) -> None:
        source = """/* Function start: 0x100 */
void first(void)
{
}

/* Function start: 0x300 */
void third(void)
{
}
"""
        candidate = """/* Function start: 0x200 */
void second(void)
{
}
"""
        updated = replace_or_insert_function(source, 0x200, candidate)
        self.assertLess(updated.index("0x100"), updated.index("0x200"))
        self.assertLess(updated.index("0x200"), updated.index("0x300"))
        self.assertEqual(current_function(updated, 0x200), candidate.rstrip())

    def test_replacing_last_function_has_one_final_newline(self) -> None:
        source = """/* Function start: 0x100 */
void before(void)
{
}

/* Function start: 0x200 */
void target(void)
{
}
"""
        candidate = """/* Function start: 0x200 */
void target(void)
{
    work();
}"""
        updated = replace_or_insert_function(source, 0x200, candidate)
        self.assertTrue(updated.endswith("    work();\n}\n"))
        self.assertFalse(updated.endswith("\n\n"))

    def test_decompilation_extraction_does_not_require_a_matching_name(self) -> None:
        decompilation = """Function: OperationalLabel
Address: 0x00401000

/* metadata containing misleading_call(foo) */
void OperationalLabel(void)

{
  first_statement();
  last_statement();
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            hint = ProjectRepository(root)._concise_decompilation(
                decompilation,
                "RecoveredFixtureName",
            )
        self.assertTrue(hint.startswith("void OperationalLabel(void)"))
        self.assertIn("last_statement();", hint)
        self.assertNotIn("metadata", hint)

    def test_candidate_schema_supports_large_functions_and_cleans_whitespace(
        self,
    ) -> None:
        candidate = Candidate(
            symbol="ComputeFixtureValue",
            prototype="int ComputeFixtureValue(void)",
            source=(
                "/* Function start: 0x401000 */  \n"
                "int ComputeFixtureValue(void)\t\n"
                "{\n" + ("    value += 1;   \n" * 900) + "    return value;\n}"
            ),
        )
        self.assertGreater(len(candidate.source), 12000)
        self.assertNotIn(";   \n", candidate.source)

    def test_marker_and_model_proposed_symbol_are_mechanical_operations(self) -> None:
        candidate = Candidate(
            symbol="ExistingFixtureAction",
            prototype="void ExistingFixtureAction(void)",
            source="""void ExistingFixtureAction(void)
{
    ExistingFixtureAction();
}""",
            supporting_insertions=[
                SupportingInsertion(
                    path="include/globals.h",
                    content="extern int g_ExistingFixtureActionCount;",
                )
            ],
        )
        marked = normalize_candidate_marker(candidate, 0x00401000)
        renamed = rename_candidate_symbol(marked, "UpdateFixtureState")
        self.assertTrue(renamed.source.startswith("/* Function start: 0x401000 */"))
        self.assertEqual(renamed.source.count("UpdateFixtureState"), 2)
        self.assertIn(
            "g_ExistingFixtureActionCount",
            renamed.supporting_insertions[0].content,
        )

    def test_complete_workspace_change_set_is_rendered_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            candidate = Candidate(
                symbol="ReadFixtureCounter",
                prototype="int ReadFixtureCounter(void)",
                source="""/* Function start: 0x401000 */
int ReadFixtureCounter(void)
{
    return g_fixture_counter_00402030.value;
}""",
                supporting_insertions=[
                    SupportingInsertion(
                        path="include/types.h",
                        content=(
                            "typedef struct FixtureCounter {\n"
                            "    int value;\n"
                            "} FixtureCounter;"
                        ),
                    ),
                    SupportingInsertion(
                        path="include/globals.h",
                        content="extern FixtureCounter g_fixture_counter_00402030;",
                    ),
                    SupportingInsertion(
                        path="src/globals.c",
                        content="FixtureCounter g_fixture_counter_00402030;",
                    ),
                ],
            )
            active = target.with_candidate_contract(candidate)
            validate_candidate(
                candidate,
                active,
                set(repository.reserved_symbols(target)),
                repository.allowed_support_paths(),
            )
            baseline = repository.snapshot_workspace(target)
            rendered = repository.render_candidate_workspace(
                active,
                candidate,
                baseline,
            )
            self.assertEqual(
                (root / "src/sample.c").read_text(encoding="utf-8"),
                '#include "project.h"\n',
            )
            self.assertIn(
                "int ReadFixtureCounter(void); /* 0x00401000 */",
                rendered[(root / "include/functions.h").resolve()],
            )
            self.assertLess(
                rendered[(root / "include/types.h").resolve()].index("FixtureCounter"),
                rendered[(root / "include/types.h").resolve()].index("#endif"),
            )

    def test_unsafe_support_insertions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            candidate = Candidate(
                symbol="ReadFixtureValue",
                prototype="int ReadFixtureValue(void)",
                source="""/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return 7;
}""",
                supporting_insertions=[
                    SupportingInsertion(
                        path="Makefile",
                        content="unexpected edit",
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "not configured"):
                validate_candidate(
                    candidate,
                    target.with_candidate_contract(candidate),
                    set(),
                    repository.allowed_support_paths(),
                )

    def test_fingerprint_includes_supporting_insertions(self) -> None:
        common = {
            "symbol": "ReadFixtureValue",
            "prototype": "int ReadFixtureValue(void)",
            "source": """/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return g_fixture_value;
}""",
        }
        first = Candidate(
            **common,
            supporting_insertions=[
                SupportingInsertion(
                    path="include/globals.h",
                    content="extern int g_fixture_value;",
                )
            ],
        )
        second = Candidate(
            **common,
            supporting_insertions=[
                SupportingInsertion(
                    path="include/globals.h",
                    content="extern short g_fixture_value;",
                )
            ],
        )
        self.assertNotEqual(candidate_fingerprint(first), candidate_fingerprint(second))

    def test_feedback_distinguishes_compiler_errors_and_compacts_assembly(self) -> None:
        compiler = "sample.c(4) : error C2065: 'missing' : undeclared identifier"
        comparison = """Comparison for function ReadFixtureValue
0x401000: MOV EAX,0x500000 | 0x500000: MOV EAX,0x600000
0x401005: ADD EAX,1 | 0x500005: SUB EAX,1
Similarity: 70.00%
"""
        self.assertTrue(ProjectRepository.has_compiler_errors(compiler))
        self.assertEqual(ProjectRepository.compiler_error_count(compiler), 1)
        compact = ProjectRepository.compact_similarity_feedback(comparison)
        self.assertIn("ADD EAX,1", compact)
        self.assertNotIn("MOV EAX", compact)
        self.assertIn("Similarity: 70.00%", compact)


class SeedAndPatchTests(unittest.TestCase):
    def test_inferred_contract_requires_names_for_non_void_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have a name"):
            ContractProposal(
                symbol="ReadFixtureValue",
                prototype="int ReadFixtureValue(int, const char *)",
            )

    def test_contract_and_decompiler_seed_are_normalized_mechanically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            write_fixture(
                root,
                "include/types.h",
                """#ifndef FIXTURE_TYPES_H
#define FIXTURE_TYPES_H

typedef struct tagRECT {
    long left;
} RECT;

#endif
""",
            )
            repository = ProjectRepository(root)
            contract = normalize_contract(
                ContractProposal(
                    symbol="CalculateFixtureTotal",
                    prototype=(
                        "bool CalculateFixtureTotal(int count, tagRECT *bounds)"
                    ),
                ),
                repository,
            )
            source = """undefined4 FUN_00401000(int param_1, tagRECT *param_2)
{
    bool result;
    ExistingFixtureAction();
    result = DAT_00402010[param_1] + DAT_00409999;
    return result;
}
"""
            normalized, changes = normalize_decompiler_seed(
                source,
                repository,
                contract,
                excluded_address=0x00401000,
            )
            candidate = candidate_from_seed(
                normalized,
                contract,
                0x00401000,
            )
            self.assertEqual(
                candidate.prototype,
                "int CalculateFixtureTotal(int count, RECT *bounds)",
            )
            self.assertIn("g_fixture_value_00402010[count]", candidate.source)
            self.assertIn("(*(int *)0x00409999)", candidate.source)
            self.assertIn("int result;", candidate.source)
            self.assertNotIn("param_1", candidate.source)
            self.assertTrue(
                any("parameter param_1 -> count" in item for item in changes)
            )

    def test_target_address_is_excluded_from_declaration_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            mapping = declaration_address_symbols(
                repository,
                excluded_address=0x00401000,
            )
            self.assertNotIn(0x00401000, mapping)
            self.assertEqual(mapping[0x00402000], "ExistingFixtureAction")

    def test_sanitizer_keeps_independent_operations_when_exact_text_is_stale(
        self,
    ) -> None:
        source = """/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return missing_value;
}"""
        patch_value = SourcePatch(
            identifier_replacements=[
                IdentifierReplacement(old="missing_value", new="fixture_value")
            ],
            edits=[ExactEdit(old="return stale_value;", new="return 7;")],
        )
        sanitized, rejected = sanitize_source_patch(
            source,
            patch_value,
            "ReadFixtureValue",
        )
        self.assertEqual(len(sanitized.identifier_replacements), 1)
        self.assertEqual(sanitized.edits, [])
        self.assertIn("old text", rejected[0])

    def test_sanitizer_rejects_brace_changes_and_operational_names(self) -> None:
        source = """/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return 7;
}"""
        raw = SourcePatch(
            edits=[ExactEdit(old="return 7;", new="{ return DAT_00402010;")]
        )
        sanitized, rejected = sanitize_source_patch(
            source,
            raw,
            "ReadFixtureValue",
        )
        self.assertEqual(sanitized.edits, [])
        self.assertTrue(rejected)

    def test_patch_application_reasserts_the_locked_header(self) -> None:
        candidate = Candidate(
            symbol="ReadFixtureValue",
            prototype="int ReadFixtureValue(void)",
            source="""/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return 7;
}""",
        )
        repaired = apply_source_patch(
            candidate,
            SourcePatch(edits=[ExactEdit(old="return 7;", new="return 8;")]),
            0x00401000,
        )
        self.assertEqual(repaired.prototype, candidate.prototype)
        self.assertIn("return 8;", repaired.source)
        self.assertEqual(repaired.source.count("Function start:"), 1)


class StagedSearchTests(unittest.TestCase):
    def test_explicit_contract_builds_a_seed_without_contract_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(
                0x00401000,
                symbol="ReadFixtureValue",
                prototype="int ReadFixtureValue(void)",
            )

            def compare(*args: object):
                self.assertIn(
                    "int ReadFixtureValue(void)",
                    target.source_path.read_text(encoding="utf-8"),
                )
                return 84.0, "Similarity: 84.00%"

            repository.compare = compare  # type: ignore[method-assign]
            with (
                patch(
                    "binary_recons.search.ManagedLlamaServer",
                    side_effect=AssertionError("the fixed contract needs no model"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ReconstructionSearch(
                    repository,
                    target,
                    SearchConfig(seed=target.address),
                    LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN),
                ).run()
            self.assertEqual(result.score, 84.0)
            seed = json.loads(
                (result.session_directory / "seed.json").read_text(encoding="utf-8")
            )
            self.assertEqual(seed["origin"], "configured-contract")

    def test_mechanical_ghidra_seed_can_reach_target_without_a_body_request(
        self,
    ) -> None:
        class FakeClient:
            body_requests = 0

            def __init__(self, *args: object):
                pass

            def infer_contract(self, prompt: str):
                return (
                    ContractProposal(
                        symbol="ReadFixtureValue",
                        prototype="int ReadFixtureValue(void)",
                    ),
                    {"kind": "contract"},
                )

            def propose_symbols(self, *args: object):
                raise AssertionError("the contract name is valid")

            def repair_compile(self, *args: object):
                type(self).body_requests += 1
                raise AssertionError("the mechanical seed already meets the target")

            def improve_similarity(self, *args: object):
                type(self).body_requests += 1
                raise AssertionError("the mechanical seed already meets the target")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)

            def compare(*args: object):
                source = target.source_path.read_text(encoding="utf-8")
                self.assertIn("int ReadFixtureValue(void)", source)
                self.assertIn("return 7;", source)
                return 84.0, "Similarity: 84.00%"

            repository.compare = compare  # type: ignore[method-assign]
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeManagedServer),
                patch("binary_recons.search.StructuredModelClient", FakeClient),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ReconstructionSearch(
                    repository,
                    target,
                    SearchConfig(seed=target.address, max_edits=0),
                    LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN),
                ).run()

            self.assertTrue(result.target_reached)
            self.assertEqual(result.score, 84.0)
            self.assertEqual(result.attempts, 1)
            self.assertEqual(FakeClient.body_requests, 0)
            self.assertIn(
                "int ReadFixtureValue(void)",
                target.source_path.read_text(encoding="utf-8"),
            )
            manifest = json.loads(
                (result.session_directory / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["workflow"], "ghidra-seed-bounded-qwen-edits-v1")
            self.assertEqual(manifest["stop_reason"], "target-reached")

    def test_compiler_failure_is_given_to_qwen_before_the_source(self) -> None:
        class FakeClient:
            compile_prompts: list[str] = []

            def __init__(self, *args: object):
                pass

            def infer_contract(self, prompt: str):
                return (
                    ContractProposal(
                        symbol="ReadFixtureValue",
                        prototype="int ReadFixtureValue(void)",
                    ),
                    {},
                )

            def propose_symbols(self, *args: object):
                raise AssertionError("unexpected name repair")

            def repair_compile(self, prompt: str, round_number: int):
                self.compile_prompts.append(prompt)
                return (
                    SourcePatch(
                        edits=[
                            ExactEdit(old="unknown_fixture_value", new="7", mode="all")
                        ]
                    ),
                    {},
                )

            def improve_similarity(self, *args: object):
                raise AssertionError("compile repair reaches the target")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root, "return unknown_fixture_value;")
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            compare_calls = 0

            def compare(*args: object):
                nonlocal compare_calls
                compare_calls += 1
                source = target.source_path.read_text(encoding="utf-8")
                if "unknown_fixture_value" in source:
                    return None, (
                        "sample.c(4) : error C2065: 'unknown_fixture_value' : "
                        "undeclared identifier"
                    )
                return 84.0, "Similarity: 84.00%"

            repository.compare = compare  # type: ignore[method-assign]
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeManagedServer),
                patch("binary_recons.search.StructuredModelClient", FakeClient),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ReconstructionSearch(
                    repository,
                    target,
                    SearchConfig(seed=target.address, max_edits=2),
                    LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN),
                ).run()

            self.assertTrue(result.target_reached)
            self.assertEqual(compare_calls, 2)
            self.assertEqual(len(FakeClient.compile_prompts), 1)
            prompt = FakeClient.compile_prompts[0]
            self.assertLess(prompt.index("error C2065"), prompt.index("CURRENT SOURCE"))
            self.assertIn("return 7;", target.source_path.read_text(encoding="utf-8"))
            edit_result = json.loads(
                (result.session_directory / "edit-01.compile.result.json").read_text()
            )
            self.assertTrue(edit_result["accepted"])

    def test_non_improving_similarity_edit_is_rolled_back(self) -> None:
        class FakeClient:
            def __init__(self, *args: object):
                pass

            def infer_contract(self, prompt: str):
                return (
                    ContractProposal(
                        symbol="ReadFixtureValue",
                        prototype="int ReadFixtureValue(void)",
                    ),
                    {},
                )

            def propose_symbols(self, *args: object):
                raise AssertionError("unexpected name repair")

            def repair_compile(self, *args: object):
                raise AssertionError("the seed compiles")

            def improve_similarity(self, prompt: str, round_number: int):
                if "Similarity: 70.00%" not in prompt:
                    raise AssertionError("similarity feedback was not supplied")
                return (
                    SimilarityPatch(edit=ExactEdit(old="return 7;", new="return 8;")),
                    {},
                )

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)

            def compare(*args: object):
                source = target.source_path.read_text(encoding="utf-8")
                score = 60.0 if "return 8;" in source else 70.0
                return score, (
                    "Comparison for function ReadFixtureValue\n"
                    "0x401000: ADD EAX,1 | 0x501000: SUB EAX,1\n"
                    "Similarity: %.2f%%" % score
                )

            repository.compare = compare  # type: ignore[method-assign]
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeManagedServer),
                patch("binary_recons.search.StructuredModelClient", FakeClient),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ReconstructionSearch(
                    repository,
                    target,
                    SearchConfig(seed=target.address, max_edits=1),
                    LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN),
                ).run()

            self.assertEqual(result.score, 70.0)
            self.assertFalse(result.target_reached)
            source = target.source_path.read_text(encoding="utf-8")
            self.assertIn("return 7;", source)
            self.assertNotIn("return 8;", source)
            decision = json.loads(
                (
                    result.session_directory / "edit-01.similarity.result.json"
                ).read_text()
            )
            self.assertFalse(decision["accepted"])

    def test_colliding_contract_name_gets_a_small_name_only_repair(self) -> None:
        class FakeClient:
            symbol_calls = 0

            def __init__(self, *args: object):
                pass

            def infer_contract(self, prompt: str):
                return (
                    ContractProposal(
                        symbol="ExistingFixtureAction",
                        prototype="int ExistingFixtureAction(void)",
                    ),
                    {},
                )

            def propose_symbols(self, prompt: str, attempt: int):
                type(self).symbol_calls += 1
                return (
                    SymbolProposalBatch(
                        symbols=[
                            "ExistingFixtureAction",
                            "DialogProc",
                            "MeasureFixtureState",
                            "ReadFixtureState",
                            "CheckFixtureState",
                            "UpdateFixtureState",
                        ]
                    ),
                    {},
                )

            def repair_compile(self, *args: object):
                raise AssertionError("seed reaches target")

            def improve_similarity(self, *args: object):
                raise AssertionError("seed reaches target")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            repository.compare = (  # type: ignore[method-assign]
                lambda *args: (84.0, "Similarity: 84.00%")
            )
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeManagedServer),
                patch("binary_recons.search.StructuredModelClient", FakeClient),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ReconstructionSearch(
                    repository,
                    target,
                    SearchConfig(seed=target.address),
                    LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN),
                ).run()
            self.assertEqual(result.symbol, "MeasureFixtureState")
            self.assertEqual(FakeClient.symbol_calls, 1)
            self.assertEqual(result.attempts, 1)
            self.assertTrue(
                (result.session_directory / "contract-name-repair-01.json").exists()
            )

    def test_selected_change_set_can_be_resumed_without_starting_llama(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            selected = root / "selected-change-set.json"
            candidate = Candidate(
                symbol="ReadFixtureValue",
                prototype="int ReadFixtureValue(void)",
                source="""/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return 7;
}""",
            )
            selected.write_text(
                json.dumps({"candidate": candidate.model_dump(mode="json")}),
                encoding="utf-8",
            )

            def compare(repository_self: object, *args: object):
                return 84.0, "Similarity: 84.00%"

            with (
                patch.object(ProjectRepository, "compare", compare),
                patch(
                    "binary_recons.search.ManagedLlamaServer",
                    side_effect=AssertionError("llama must not start"),
                ),
                patch("binary_recons.cli.DEFAULT_MODEL_PATH", None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = main(
                    [
                        "--project-root",
                        str(root),
                        "--address",
                        "0x401000",
                        "--resume-candidate",
                        str(selected),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertIn(
                "int ReadFixtureValue(void)",
                (root / "src/sample.c").read_text(encoding="utf-8"),
            )

    def test_model_failure_after_a_compiled_baseline_preserves_that_baseline(
        self,
    ) -> None:
        class FakeClient:
            def __init__(self, *args: object):
                pass

            def improve_similarity(self, *args: object):
                raise ModelRequestError("synthetic timeout")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            write_fixture(
                root,
                "src/sample.c",
                """/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return 7;
}
""",
            )
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            repository.compare = (  # type: ignore[method-assign]
                lambda *args: (70.0, "Similarity: 70.00%")
            )
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeManagedServer),
                patch("binary_recons.search.StructuredModelClient", FakeClient),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ReconstructionSearch(
                    repository,
                    target,
                    SearchConfig(seed=target.address, max_edits=1),
                    LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN),
                ).run()
            self.assertEqual(result.score, 70.0)
            self.assertIn("return 7;", target.source_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (result.session_directory / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["stop_reason"],
                "model-request-failed-after-compiled-seed",
            )

    def test_unexpected_compare_failure_restores_every_workspace_file(self) -> None:
        class FakeClient:
            def __init__(self, *args: object):
                pass

            def infer_contract(self, prompt: str):
                return (
                    ContractProposal(
                        symbol="ReadFixtureValue",
                        prototype="int ReadFixtureValue(void)",
                    ),
                    {},
                )

            def propose_symbols(self, *args: object):
                raise AssertionError("unexpected name repair")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            baseline = repository.snapshot_workspace(target)

            def interrupted_compare(*args: object):
                self.assertIn(
                    "ReadFixtureValue",
                    target.source_path.read_text(encoding="utf-8"),
                )
                raise RuntimeError("synthetic build interruption")

            repository.compare = interrupted_compare  # type: ignore[method-assign]
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeManagedServer),
                patch("binary_recons.search.StructuredModelClient", FakeClient),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "synthetic build interruption"),
            ):
                ReconstructionSearch(
                    repository,
                    target,
                    SearchConfig(seed=target.address),
                    LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN),
                ).run()
            self.assertEqual(repository.snapshot_workspace(target), baseline)


class StructuredClientTests(unittest.TestCase):
    def test_qwen_requests_use_json_schema_non_thinking_and_bounded_sampling(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        class FakeCompletions:
            def create_with_completion(self, **kwargs: object):
                calls.append(kwargs)
                response_model = kwargs["response_model"]
                if response_model is ContractProposal:
                    return (
                        ContractProposal(
                            symbol="ReadFixtureValue",
                            prototype="int ReadFixtureValue(void)",
                        ),
                        {},
                    )
                return (
                    SymbolProposalBatch(
                        symbols=[
                            "ReadFixtureValue",
                            "MeasureFixtureValue",
                            "CheckFixtureValue",
                            "UpdateFixtureValue",
                            "ComputeFixtureValue",
                            "SelectFixtureValue",
                        ]
                    ),
                    {},
                )

        class FakeInstructor:
            class Chat:
                completions = FakeCompletions()

            chat = Chat()

        class FakeOpenAI:
            def close(self) -> None:
                pass

        class FakeServer:
            base_url = "http://127.0.0.1:8080/v1"
            config = LlamaServerConfig(
                model_path=None,
                preset=ModelPreset.QWEN,
            )

            def ensure_alive(self) -> None:
                pass

        with (
            patch("binary_recons.model_client.OpenAI", return_value=FakeOpenAI()),
            patch(
                "binary_recons.model_client.instructor.from_openai",
                return_value=FakeInstructor(),
            ),
        ):
            client = StructuredModelClient(
                FakeServer(),  # type: ignore[arg-type]
                SearchConfig(seed=1, max_tokens=320),
            )
            client.infer_contract("contract prompt")
            client.propose_symbols("name prompt", 1)
            client.close()

        contract_call, naming_call = calls
        self.assertEqual(contract_call["max_tokens"], 192)
        self.assertEqual(contract_call["temperature"], 0.2)
        self.assertEqual(contract_call["presence_penalty"], 0.0)
        contract_extra = contract_call["extra_body"]
        self.assertEqual(contract_extra["top_k"], 20)  # type: ignore[index]
        self.assertEqual(
            contract_extra["chat_template_kwargs"],  # type: ignore[index]
            {"enable_thinking": False},
        )
        self.assertEqual(naming_call["max_tokens"], 160)
        self.assertEqual(naming_call["temperature"], 0.7)
        self.assertEqual(naming_call["presence_penalty"], 1.5)


class LlamaServerTests(unittest.TestCase):
    def search_config(self) -> SearchConfig:
        return SearchConfig(seed=1, max_edits=1)

    def test_managed_server_is_started_monitored_and_stopped(self) -> None:
        port = unused_port()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = LlamaServerConfig(
                port=port,
                startup_timeout=5,
                shutdown_timeout=2,
                health_interval=0.02,
                command_override=[sys.executable, "-u", "-c", FAKE_SERVER, str(port)],
            )
            server = ManagedLlamaServer(config, self.search_config(), directory)
            with server:
                self.assertIsNotNone(server.process)
                assert server.process is not None
                pid = server.process.pid
                self.assertIsNone(server.process.poll())
                server.ensure_alive()
            self.assertIsNotNone(server.process.poll())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            manifest = json.loads((directory / "llama-server.json").read_text())
            self.assertEqual(manifest["status"], "stopped")
            self.assertTrue(manifest["owned"])

    def test_external_server_is_never_stopped(self) -> None:
        port = unused_port()
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", FAKE_SERVER, str(port)],
            start_new_session=True,
        )
        try:
            wait_ready(port)
            with tempfile.TemporaryDirectory() as temporary:
                config = LlamaServerConfig(mode=ServerMode.EXTERNAL, port=port)
                server = ManagedLlamaServer(
                    config,
                    self.search_config(),
                    Path(temporary),
                )
                with server:
                    server.ensure_alive()
                self.assertIsNone(process.poll())
                manifest = json.loads(
                    (Path(temporary) / "llama-server.json").read_text()
                )
                self.assertEqual(manifest["status"], "released")
                self.assertFalse(manifest["owned"])
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)

    def test_managed_server_stops_when_work_raises(self) -> None:
        port = unused_port()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = LlamaServerConfig(
                port=port,
                startup_timeout=5,
                shutdown_timeout=2,
                health_interval=0.02,
                command_override=[sys.executable, "-u", "-c", FAKE_SERVER, str(port)],
            )
            server = ManagedLlamaServer(config, self.search_config(), directory)
            with self.assertRaisesRegex(RuntimeError, "expected test failure"):
                with server:
                    raise RuntimeError("expected test failure")
            assert server.process is not None
            self.assertIsNotNone(server.process.poll())
            manifest = json.loads((directory / "llama-server.json").read_text())
            self.assertEqual(manifest["status"], "stopped")
            self.assertEqual(manifest["stop_reason"], "exception")

    def test_model_presets_keep_qwen_flags_out_of_gemma(self) -> None:
        search = SearchConfig(seed=1, max_edits=1)
        qwen = LlamaServerConfig(
            binary=Path("llama-server"),
            model_path=Path("qwen.gguf"),
            preset=ModelPreset.QWEN,
        )
        gemma = LlamaServerConfig(
            binary=Path("llama-server"),
            model_path=Path("gemma.gguf"),
            preset=ModelPreset.GEMMA,
        )
        qwen_command = qwen.command(search)
        gemma_command = gemma.command(search)
        self.assertIn("draft-mtp", qwen_command)
        self.assertNotIn("draft-mtp", gemma_command)
        self.assertEqual(qwen.resolved_preset(), ModelPreset.QWEN)
        self.assertEqual(
            gemma_command[gemma_command.index("--top-k") + 1],
            "64",
        )


if __name__ == "__main__":
    unittest.main()

"""Fast tests for the reconstruction package and server ownership boundary."""

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

from binary_recons.cli import entrypoint, main  # noqa: E402
from binary_recons.llama_server import ManagedLlamaServer  # noqa: E402
from binary_recons.model_client import ModelRequestError  # noqa: E402
from binary_recons.models import (  # noqa: E402
    Candidate,
    CandidateBatch,
    LlamaServerConfig,
    ModelPreset,
    SearchConfig,
    ServerMode,
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


def make_fixture_project(root: Path) -> None:
    write_fixture(
        root,
        "binary-recons.toml",
        'language = "C"\n'
        'compiler = "Fixture Compiler 1.0"\n'
        'exports_dir = "analysis"\n'
        'output_dir = "artifacts/reconstruction"\n'
        'strings_file = "analysis/strings.txt"\n'
        'source_dirs = ["src"]\n'
        'declaration_files = ["include/*.h"]\n'
        'prototype_file = "include/functions.h"\n'
        'rule_profiles = ["c89"]\n'
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
        "int sample_function(void); /* 0x00401000 */\n"
        "void ExistingHelper(void); /* 0x00402000 */\n",
    )
    write_fixture(
        root,
        "include/globals.h",
        "#ifndef FIXTURE_GLOBALS_H\n"
        "#define FIXTURE_GLOBALS_H\n\n"
        "extern int g_sample_value_00402010[4];\n"
        "extern SampleVector g_sample_vector_00402020[4];\n\n"
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
        "{\n"
        "  return 7;\n"
        "}\n",
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

    def test_new_target_contract_is_left_for_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            target = ProjectRepository(root).resolve_target(0x00401000)
            self.assertIsNone(target.symbol)
            self.assertEqual(target.source_path, (root / "src/sample.c").resolve())
            self.assertIsNone(target.prototype)
            self.assertEqual(
                ProjectRepository(root).reserved_symbols(target), ["ExistingHelper"]
            )

    def test_cli_dry_run_uses_project_root(self) -> None:
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
                    "*/iteration-01.prompt.txt"
                )
            )
            self.assertEqual(len(prompts), 1)
            prompt = prompts[0].read_text(encoding="utf-8")
            self.assertIn("[shared profile: c89]", prompt)
            self.assertIn("[project file: RECONSTRUCTION.md]", prompt)
            self.assertIn('0x00403000: "fixture text"', prompt)
            self.assertIn("No source-level name or interface is supplied", prompt)
            self.assertNotIn("int sample_function(void)", prompt)
            self.assertIn("ExistingHelper", prompt)
            self.assertIn("supporting_insertions", prompt)
            self.assertIn(
                "[allowed support file: include/types.h]",
                prompt,
            )

    def test_existing_definition_keeps_its_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            write_fixture(
                root,
                "src/sample.c",
                """/* Function start: 0x401000 */
int EstablishedName(void)
{
    return 7;
}
""",
            )
            target = ProjectRepository(root).resolve_target(0x00401000)
            self.assertEqual(target.symbol, "EstablishedName")
            self.assertEqual(target.prototype, "int EstablishedName(void)")

    def test_reopen_contract_hides_an_existing_weak_interface(self) -> None:
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
            prompts = list(
                (root / "artifacts/reconstruction/local-model/00401000").glob(
                    "*/iteration-01.prompt.txt"
                )
            )
            self.assertEqual(len(prompts), 1)
            prompt = prompts[0].read_text(encoding="utf-8")
            self.assertIn("No source-level name or interface is supplied", prompt)
            self.assertNotIn("DialogProc", prompt)

    def test_bare_mechanism_function_name_is_rejected(self) -> None:
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
                    ["DialogProc", "ScorePanelDialogProc"], set(), 0x00401000
                ),
                "ScorePanelDialogProc",
            )

    def test_insertion_is_address_sorted(self) -> None:
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

    def test_replacing_last_function_does_not_add_blank_line_at_eof(self) -> None:
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

    def test_decompilation_extraction_does_not_require_matching_name(self) -> None:
        decompilation = """Function: OperationalLabel
Address: 0x00401000

/* metadata containing a misleading call(foo) */
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
                decompilation, "RecoveredDeveloperName"
            )
        self.assertTrue(hint.startswith("void OperationalLabel(void)"))
        self.assertIn("first_statement();", hint)
        self.assertIn("last_statement();", hint)
        self.assertNotIn("metadata", hint)

    def test_decompiler_alias_gets_address_matched_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            evidence = ProjectRepository(root)._declaration_evidence(
                "DAT_00402010[0] = g_pSampleVector_00402020[0].x;"
            )
        self.assertIn("g_sample_value_00402010", evidence)
        self.assertIn("g_sample_vector_00402020", evidence)

    def test_comparison_command_is_project_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            self.assertEqual(
                repository.config.comparison_command(
                    "ComputeFixtureValue", target.address
                ),
                ["fixture-compare", "ComputeFixtureValue", "00401000"],
            )

    def test_candidate_schema_accepts_large_functions(self) -> None:
        candidate = Candidate(
            symbol="ComputeFixtureValue",
            prototype="int ComputeFixtureValue(void)",
            source="x" * 6000,
        )
        self.assertEqual(len(candidate.source), 6000)

    def test_candidate_schema_removes_line_end_whitespace(self) -> None:
        candidate = Candidate(
            symbol="ComputeFixtureValue",
            prototype="int ComputeFixtureValue(void)",
            source=(
                "/* Function start: 0x401000 */  \n"
                "int ComputeFixtureValue(void)\t\n"
                "{\n"
                "    return 7;   \n"
                "}"
            ),
            supporting_insertions=[
                SupportingInsertion(
                    path="include/globals.h",
                    content="extern int g_fixture;  \n\n",
                )
            ],
        )

        self.assertEqual(
            candidate.source,
            "/* Function start: 0x401000 */\n"
            "int ComputeFixtureValue(void)\n"
            "{\n"
            "    return 7;\n"
            "}",
        )
        self.assertEqual(
            candidate.supporting_insertions[0].content,
            "extern int g_fixture;",
        )

    def test_missing_address_marker_is_added_mechanically(self) -> None:
        candidate = Candidate(
            symbol="ComputeFixtureValue",
            prototype="int ComputeFixtureValue(void)",
            source="""int ComputeFixtureValue(void)
{
    return 7;
}""",
        )
        normalized = normalize_candidate_marker(candidate, 0x00401000)
        self.assertTrue(
            normalized.source.startswith("/* Function start: 0x401000 */\n")
        )

    def test_model_proposed_symbol_is_applied_only_to_candidate_contract(self) -> None:
        candidate = Candidate(
            symbol="ExistingHelper",
            prototype="void ExistingHelper(void)",
            source="""/* Function start: 0x401000 */
void ExistingHelper(void)
{
    ExistingHelper();
}""",
            supporting_insertions=[
                SupportingInsertion(
                    path="include/globals.h",
                    content="extern int g_ExistingHelperCount;",
                )
            ],
        )

        renamed = rename_candidate_symbol(candidate, "ComputeFixtureValue")

        self.assertEqual(renamed.symbol, "ComputeFixtureValue")
        self.assertEqual(renamed.prototype, "void ComputeFixtureValue(void)")
        self.assertEqual(renamed.source.count("ComputeFixtureValue"), 2)
        self.assertEqual(
            renamed.supporting_insertions[0].content,
            "extern int g_ExistingHelperCount;",
        )

    def test_complete_change_set_renders_support_and_prototype_together(self) -> None:
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
                        content=("extern FixtureCounter g_fixture_counter_00402030;"),
                    ),
                    SupportingInsertion(
                        path="src/globals.c",
                        content="FixtureCounter g_fixture_counter_00402030;",
                    ),
                    SupportingInsertion(
                        path="include/functions.h",
                        content="void InspectFixtureCounter(int index);",
                    ),
                ],
            )
            candidate_target = target.with_candidate_contract(candidate)
            validate_candidate(
                candidate,
                candidate_target,
                set(repository.reserved_symbols(target)),
                repository.allowed_support_paths(),
            )
            baseline = repository.snapshot_workspace(target)
            rendered = repository.render_candidate_workspace(
                candidate_target, candidate, baseline
            )

            self.assertEqual(
                (root / "src/sample.c").read_text(encoding="utf-8"),
                '#include "project.h"\n',
            )
            self.assertIn(
                "int ReadFixtureCounter(void); /* 0x00401000 */",
                rendered[(root / "include/functions.h").resolve()],
            )
            self.assertTrue(
                rendered[(root / "include/functions.h").resolve()].endswith(
                    "void InspectFixtureCounter(int index);\n"
                )
            )
            types = rendered[(root / "include/types.h").resolve()]
            self.assertLess(types.index("FixtureCounter"), types.index("#endif"))
            self.assertIn(
                "extern FixtureCounter g_fixture_counter_00402030;",
                rendered[(root / "include/globals.h").resolve()],
            )
            self.assertTrue(
                rendered[(root / "src/globals.c").resolve()].endswith(
                    "FixtureCounter g_fixture_counter_00402030;\n"
                )
            )

    def test_target_prototype_cannot_be_supplied_as_supporting_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            candidate = Candidate(
                symbol="ComputeFixtureValue",
                prototype="int ComputeFixtureValue(void)",
                source="""/* Function start: 0x401000 */
int ComputeFixtureValue(void)
{
    return 7;
}""",
                supporting_insertions=[
                    SupportingInsertion(
                        path="include/functions.h",
                        content="int ComputeFixtureValue(void);",
                    )
                ],
            )

            with self.assertRaisesRegex(
                ValueError, "target prototype belongs in the managed prototype file"
            ):
                validate_candidate(
                    candidate,
                    target.with_candidate_contract(candidate),
                    set(repository.reserved_symbols(target)),
                    repository.allowed_support_paths(),
                )

    def test_supporting_content_cannot_redeclare_an_existing_function(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            candidate = Candidate(
                symbol="ComputeFixtureValue",
                prototype="int ComputeFixtureValue(void)",
                source="""/* Function start: 0x401000 */
int ComputeFixtureValue(void)
{
    return 7;
}""",
                supporting_insertions=[
                    SupportingInsertion(
                        path="include/functions.h",
                        content="int ExistingHelper(int invented_parameter);",
                    )
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "support insertion redeclares existing function: ExistingHelper",
            ):
                validate_candidate(
                    candidate,
                    target.with_candidate_contract(candidate),
                    set(repository.reserved_symbols(target)),
                    repository.allowed_support_paths(),
                )

    def test_applying_unchanged_workspace_preserves_file_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            workspace = repository.snapshot_workspace(target)
            with patch("binary_recons.repository.atomic_write") as writer:
                repository.apply_workspace(workspace)
            writer.assert_not_called()

    def test_unconfigured_support_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            candidate = Candidate(
                symbol="ComputeFixtureValue",
                prototype="int ComputeFixtureValue(void)",
                source="""/* Function start: 0x401000 */
int ComputeFixtureValue(void)
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
            "symbol": "ComputeFixtureValue",
            "prototype": "int ComputeFixtureValue(void)",
            "source": """/* Function start: 0x401000 */
int ComputeFixtureValue(void)
{
    return g_fixture_value_00402030;
}""",
        }
        first = Candidate(
            **common,
            supporting_insertions=[
                SupportingInsertion(
                    path="include/globals.h",
                    content="extern int g_fixture_value_00402030;",
                )
            ],
        )
        second = Candidate(
            **common,
            supporting_insertions=[
                SupportingInsertion(
                    path="include/globals.h",
                    content="extern short g_fixture_value_00402030;",
                )
            ],
        )
        self.assertNotEqual(candidate_fingerprint(first), candidate_fingerprint(second))

    def test_compiler_errors_are_distinguished_from_compare_failures(self) -> None:
        self.assertTrue(
            ProjectRepository.has_compiler_errors(
                "sample.c(4) : error C2065: 'value' : undeclared identifier"
            )
        )
        self.assertTrue(
            ProjectRepository.has_compiler_errors(
                "sample.c:4:12: error: use of undeclared identifier 'value'"
            )
        )
        self.assertFalse(
            ProjectRepository.has_compiler_errors("BUILD/COMPARE TIMED OUT")
        )


class ValidationRepairTests(unittest.TestCase):
    def test_reserved_name_rejection_gets_a_focused_contract_repair(self) -> None:
        rejected = Candidate(
            symbol="ExistingHelper",
            prototype="void ExistingHelper(void)",
            source="""/* Function start: 0x401000 */
void ExistingHelper(void)
{
}""",
        )

        class FakeServer:
            def __init__(self, *args: object):
                pass

            def __enter__(self) -> FakeServer:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        class FakeGenerator:
            repair_prompts: list[str] = []

            def __init__(self, *args: object):
                pass

            def generate(
                self, prompt: str, iteration: int
            ) -> tuple[CandidateBatch, dict[str, object]]:
                return CandidateBatch(candidates=[rejected]), {}

            def propose_symbols(
                self,
                prompt: str,
                iteration: int,
                candidate_index: int,
                repair_attempt: int,
            ) -> tuple[SymbolProposalBatch, dict[str, object]]:
                self.repair_prompts.append(prompt)
                return (
                    SymbolProposalBatch(
                        symbols=[
                            "ExistingHelper",
                            "ComputeFixtureValue",
                            "EvaluateFixtureState",
                        ]
                    ),
                    {},
                )

            def repair(self, *args: object) -> None:
                raise AssertionError("full candidate repair should not be requested")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)

            def compare_candidate(*args: object) -> tuple[float | None, str]:
                source = target.source_path.read_text(encoding="utf-8")
                self.assertIn("void ComputeFixtureValue(void)", source)
                self.assertNotIn("void ExistingHelper(void)\n{", source)
                return 100.0, "Similarity: 100.00%"

            repository.compare = compare_candidate  # type: ignore[method-assign]
            config = SearchConfig(
                seed=1,
                max_iterations=1,
                candidates_per_iteration=1,
                compile_repair_attempts=1,
            )
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeServer),
                patch("binary_recons.search.CandidateGenerator", FakeGenerator),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ReconstructionSearch(
                    repository,
                    target,
                    config,
                    LlamaServerConfig(model_path=None),
                ).run()

            self.assertTrue(result.target_reached)
            self.assertEqual(result.attempts, 2)
            self.assertEqual(result.symbol, "ComputeFixtureValue")
            self.assertEqual(len(FakeGenerator.repair_prompts), 1)
            self.assertIn(
                "candidate symbol is already used",
                FakeGenerator.repair_prompts[0],
            )
            self.assertIn("naming task only", FakeGenerator.repair_prompts[0])
            self.assertIn("ExistingHelper", FakeGenerator.repair_prompts[0])
            self.assertIn(
                "void ComputeFixtureValue(void); /* 0x00401000 */",
                (root / "include/functions.h").read_text(encoding="utf-8"),
            )
            initial_log = (
                result.session_directory / "iteration-01-candidate-01.compare.txt"
            )
            self.assertIn("REJECTED BEFORE BUILD", initial_log.read_text())
            self.assertTrue(
                (
                    result.session_directory
                    / "iteration-01-candidate-01-repair-01.symbols.json"
                ).exists()
            )


class CompileRepairTests(unittest.TestCase):
    def test_support_file_failure_and_first_repair_can_self_repair(self) -> None:
        failing_candidate = """/* Function start: 0x401000 */
int ComputeFixtureValue(void)
{
    return g_fixture_counter_00402030.value;
}"""

        def change_set(type_name: str, define_type: bool = False) -> Candidate:
            insertions = []
            if define_type:
                insertions.append(
                    SupportingInsertion(
                        path="include/types.h",
                        content=(
                            "typedef struct FixtureCounter {\n"
                            "    int value;\n"
                            "} FixtureCounter;"
                        ),
                    )
                )
            insertions.extend(
                [
                    SupportingInsertion(
                        path="include/globals.h",
                        content=("extern %s g_fixture_counter_00402030;" % type_name),
                    ),
                    SupportingInsertion(
                        path="src/globals.c",
                        content="%s g_fixture_counter_00402030;" % type_name,
                    ),
                ]
            )
            return Candidate(
                symbol="ComputeFixtureValue",
                prototype="int ComputeFixtureValue(void)",
                source=failing_candidate,
                supporting_insertions=insertions,
            )

        class FakeServer:
            def __init__(self, *args: object):
                pass

            def __enter__(self) -> FakeServer:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        class FakeGenerator:
            repair_prompts: list[str] = []
            repair_calls = 0

            def __init__(self, *args: object):
                pass

            def generate(
                self, prompt: str, iteration: int
            ) -> tuple[CandidateBatch, dict[str, object]]:
                return (
                    CandidateBatch(candidates=[change_set("UnknownFixtureType")]),
                    {"kind": "initial"},
                )

            def repair(
                self,
                prompt: str,
                iteration: int,
                candidate_index: int,
                repair_attempt: int,
            ) -> tuple[Candidate, dict[str, object]]:
                self.repair_prompts.append(prompt)
                type(self).repair_calls += 1
                if repair_attempt == 1:
                    return change_set("UnresolvedFixtureType"), {"kind": "repair-one"}
                return change_set("FixtureCounter", define_type=True), {
                    "kind": "repair-two"
                }

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)

            def compare_candidate(*args: object) -> tuple[float | None, str]:
                source = target.source_path.read_text(encoding="utf-8")
                headers = (root / "include/globals.h").read_text(encoding="utf-8")
                types = (root / "include/types.h").read_text(encoding="utf-8")
                globals_source = (root / "src/globals.c").read_text(encoding="utf-8")
                prototypes = (root / "include/functions.h").read_text(encoding="utf-8")
                if "UnknownFixtureType" in headers:
                    return (
                        None,
                        "globals.h(7) : error C2146: syntax error : missing ';' "
                        "before identifier 'g_fixture_counter_00402030'",
                    )
                if "UnresolvedFixtureType" in headers:
                    return (
                        None,
                        "globals.h(7) : error C2061: syntax error : identifier "
                        "'UnresolvedFixtureType'",
                    )
                if all(
                    (
                        "g_fixture_counter_00402030.value" in source,
                        "typedef struct FixtureCounter" in types,
                        "extern FixtureCounter g_fixture_counter_00402030;" in headers,
                        "FixtureCounter g_fixture_counter_00402030;" in globals_source,
                        "int ComputeFixtureValue(void); /* 0x00401000 */" in prototypes,
                    )
                ):
                    return 100.0, "Similarity: 100.00%"
                raise AssertionError("unexpected candidate source")

            repository.compare = compare_candidate  # type: ignore[method-assign]
            config = SearchConfig(
                seed=1,
                max_iterations=1,
                candidates_per_iteration=1,
                compile_repair_attempts=2,
            )
            server_config = LlamaServerConfig(model_path=None)
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeServer),
                patch("binary_recons.search.CandidateGenerator", FakeGenerator),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ReconstructionSearch(
                    repository, target, config, server_config
                ).run()

            self.assertTrue(result.target_reached)
            self.assertEqual(result.score, 100.0)
            self.assertEqual(result.attempts, 3)
            self.assertEqual(result.symbol, "ComputeFixtureValue")
            self.assertEqual(FakeGenerator.repair_calls, 2)
            self.assertIn("UnknownFixtureType", FakeGenerator.repair_prompts[0])
            self.assertIn("error C2146", FakeGenerator.repair_prompts[0])
            self.assertIn("UnresolvedFixtureType", FakeGenerator.repair_prompts[1])
            self.assertIn("error C2061", FakeGenerator.repair_prompts[1])
            self.assertIn(
                "g_fixture_counter_00402030.value",
                target.source_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "int ComputeFixtureValue(void); /* 0x00401000 */",
                (root / "include/functions.h").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "typedef struct FixtureCounter",
                (root / "include/types.h").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "extern FixtureCounter g_fixture_counter_00402030;",
                (root / "include/globals.h").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "FixtureCounter g_fixture_counter_00402030;",
                (root / "src/globals.c").read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "UnresolvedFixtureType",
                (root / "include/globals.h").read_text(encoding="utf-8"),
            )
            repair_logs = list(result.session_directory.glob("*-repair-01.c"))
            self.assertEqual(len(repair_logs), 1)
            second_repair_logs = list(result.session_directory.glob("*-repair-02.c"))
            self.assertEqual(len(second_repair_logs), 1)
            selected = json.loads(
                (result.session_directory / "selected-change-set.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(selected["score"], 100.0)
            self.assertEqual(
                set(selected["changed_files"]),
                {
                    "include/functions.h",
                    "include/globals.h",
                    "include/types.h",
                    "src/globals.c",
                    "src/sample.c",
                },
            )
            self.assertEqual(len(selected["candidate"]["supporting_insertions"]), 3)

    def test_duplicate_compile_repair_uses_the_remaining_attempt(self) -> None:
        failing = Candidate(
            symbol="ComputeFixtureValue",
            prototype="int ComputeFixtureValue(void)",
            source="""/* Function start: 0x401000 */
int ComputeFixtureValue(void)
{
    return missing_fixture_value;
}""",
        )
        repaired = Candidate(
            symbol="ComputeFixtureValue",
            prototype="int ComputeFixtureValue(void)",
            source="""/* Function start: 0x401000 */
int ComputeFixtureValue(void)
{
    return 7;
}""",
        )

        class FakeServer:
            def __init__(self, *args: object):
                pass

            def __enter__(self) -> FakeServer:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        class FakeGenerator:
            repair_prompts: list[str] = []

            def __init__(self, *args: object):
                pass

            def generate(
                self, prompt: str, iteration: int
            ) -> tuple[CandidateBatch, dict[str, object]]:
                return CandidateBatch(candidates=[failing]), {}

            def repair(
                self,
                prompt: str,
                iteration: int,
                candidate_index: int,
                repair_attempt: int,
            ) -> tuple[Candidate, dict[str, object]]:
                self.repair_prompts.append(prompt)
                return (failing if repair_attempt == 1 else repaired), {}

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)

            def compare_candidate(*args: object) -> tuple[float | None, str]:
                source = target.source_path.read_text(encoding="utf-8")
                if "missing_fixture_value" in source:
                    return (
                        None,
                        "sample.c(4) : error C2065: 'missing_fixture_value' : "
                        "undeclared identifier",
                    )
                return 100.0, "Similarity: 100.00%"

            repository.compare = compare_candidate  # type: ignore[method-assign]
            config = SearchConfig(
                seed=1,
                max_iterations=1,
                candidates_per_iteration=1,
                compile_repair_attempts=2,
            )
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeServer),
                patch("binary_recons.search.CandidateGenerator", FakeGenerator),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ReconstructionSearch(
                    repository,
                    target,
                    config,
                    LlamaServerConfig(model_path=None),
                ).run()

            self.assertTrue(result.target_reached)
            self.assertEqual(result.attempts, 2)
            self.assertEqual(len(FakeGenerator.repair_prompts), 2)
            self.assertTrue(
                all(
                    "missing_fixture_value" in prompt
                    for prompt in FakeGenerator.repair_prompts
                )
            )
            first_repair = (
                result.session_directory
                / "iteration-01-candidate-01-repair-01.compare.txt"
            )
            self.assertIn("fingerprint already tried", first_repair.read_text())

    def test_unexpected_build_failure_restores_every_workspace_file(self) -> None:
        candidate = Candidate(
            symbol="ComputeFixtureValue",
            prototype="int ComputeFixtureValue(void)",
            source="""/* Function start: 0x401000 */
int ComputeFixtureValue(void)
{
    return g_temporary_value_00402030;
}""",
            supporting_insertions=[
                SupportingInsertion(
                    path="include/globals.h",
                    content="extern int g_temporary_value_00402030;",
                ),
                SupportingInsertion(
                    path="src/globals.c",
                    content="int g_temporary_value_00402030;",
                ),
            ],
        )

        class FakeServer:
            def __init__(self, *args: object):
                pass

            def __enter__(self) -> FakeServer:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        class FakeGenerator:
            def __init__(self, *args: object):
                pass

            def generate(
                self, prompt: str, iteration: int
            ) -> tuple[CandidateBatch, dict[str, object]]:
                return CandidateBatch(candidates=[candidate]), {}

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            baseline = repository.snapshot_workspace(target)

            def interrupted_compare(*args: object) -> tuple[float | None, str]:
                self.assertIn(
                    "g_temporary_value_00402030",
                    target.source_path.read_text(encoding="utf-8"),
                )
                self.assertIn(
                    "extern int g_temporary_value_00402030;",
                    (root / "include/globals.h").read_text(encoding="utf-8"),
                )
                self.assertIn(
                    "int ComputeFixtureValue(void); /* 0x00401000 */",
                    (root / "include/functions.h").read_text(encoding="utf-8"),
                )
                raise RuntimeError("synthetic build interruption")

            repository.compare = interrupted_compare  # type: ignore[method-assign]
            config = SearchConfig(
                seed=1,
                max_iterations=1,
                candidates_per_iteration=1,
            )
            with (
                patch("binary_recons.search.ManagedLlamaServer", FakeServer),
                patch("binary_recons.search.CandidateGenerator", FakeGenerator),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "synthetic build interruption"),
            ):
                ReconstructionSearch(
                    repository,
                    target,
                    config,
                    LlamaServerConfig(model_path=None),
                ).run()

            self.assertEqual(repository.snapshot_workspace(target), baseline)


class LlamaServerTests(unittest.TestCase):
    def search_config(self) -> SearchConfig:
        return SearchConfig(seed=1, max_iterations=1)

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
                config = LlamaServerConfig(
                    mode=ServerMode.EXTERNAL,
                    port=port,
                )
                server = ManagedLlamaServer(
                    config, self.search_config(), Path(temporary)
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
        search = SearchConfig(seed=1, max_iterations=1)
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
        self.assertEqual(
            gemma_command[gemma_command.index("--top-k") + 1],
            "64",
        )
        self.assertEqual(
            gemma_command[gemma_command.index("--temp") + 1],
            "1.0",
        )


if __name__ == "__main__":
    unittest.main()

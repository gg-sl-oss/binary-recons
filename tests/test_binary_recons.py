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

from instructor.core.exceptions import IncompleteOutputException


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
    MODEL_ENVIRONMENT_VARIABLE,
    ModelPreset,
    SearchConfig,
    ServerMode,
    SimilarityPatch,
    SourcePatch,
    SupportingInsertion,
    SymbolProposalBatch,
    discover_default_model_path,
)
from binary_recons.repository import (  # noqa: E402
    ProjectRepository,
    current_function,
    normalize_candidate_marker,
    rename_candidate_symbol,
    replace_or_insert_function,
    source_safety_feedback,
    source_safety_violations,
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
    bind_supporting_address_symbols,
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
        "extern HGDIOBJ g_fixture_handle_00402030;\n\n"
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
    def test_default_model_path_selects_the_best_cached_qwen_first_shard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            write_fixture(
                cache,
                "models--unsloth--Qwen3.5-35B-GGUF/snapshots/old/BF16/"
                "Qwen3.5-35B-BF16.gguf",
                "old",
            )
            write_fixture(
                cache,
                "models--unsloth--Qwen3.8-27B-GGUF/snapshots/current/Q4_K_M/"
                "Qwen3.8-27B-Q4_K_M.gguf",
                "quantized",
            )
            expected = cache / (
                "models--unsloth--Qwen3.8-27B-GGUF/snapshots/current/BF16/"
                "Qwen3.8-27B-BF16-00001-of-00002.gguf"
            )
            write_fixture(cache, str(expected.relative_to(cache)), "first")
            write_fixture(
                cache,
                "models--unsloth--Qwen3.8-27B-GGUF/snapshots/current/BF16/"
                "Qwen3.8-27B-BF16-00002-of-00002.gguf",
                "second",
            )
            write_fixture(
                cache,
                "models--unsloth--Qwen3.8-27B-GGUF/snapshots/current/mmproj-BF16.gguf",
                "projector",
            )
            write_fixture(
                cache,
                "models--google--gemma-4-GGUF/snapshots/current/gemma-4-BF16.gguf",
                "other family",
            )
            write_fixture(
                cache,
                "models--unsloth--Qwen4-72B-GGUF/snapshots/incomplete/BF16/"
                "Qwen4-72B-BF16-00001-of-00002.gguf",
                "missing its second shard",
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "binary_recons.models.DEFAULT_HUGGINGFACE_HUB_CACHE",
                    cache,
                ),
            ):
                self.assertEqual(discover_default_model_path(), expected)

    def test_explicit_model_environment_override_precedes_cache(self) -> None:
        configured = Path("~/models/explicit.gguf").expanduser()
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {MODEL_ENVIRONMENT_VARIABLE: "~/models/explicit.gguf"},
                clear=True,
            ):
                self.assertEqual(
                    discover_default_model_path(Path(temporary)),
                    configured,
                )

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
                        "--next-function",
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
        parser = build_parser()
        options = {
            option for action in parser._actions for option in action.option_strings
        }
        self.assertNotIn("--draft-source", options)
        self.assertIn("--max-edits", options)
        self.assertIn("--max-iterations", options)
        self.assertIn("--next-function", options)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args([])
        args = parser.parse_args(["--address", "0x401000"])
        self.assertFalse(args.next_function)
        self.assertEqual(args.max_callees, 2)
        self.assertEqual(args.max_tokens, 768)
        self.assertTrue(parser.parse_args(["--next-function"]).next_function)

    def test_next_target_uses_only_complete_exports_in_source_unit_ranges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            write_fixture(
                root,
                "src/sample.c",
                "/* Function start: 0x401000 */\n"
                "int ExistingFunction(void)\n"
                "{\n"
                "    return 0;\n"
                "}\n",
            )
            write_fixture(
                root,
                "analysis/FUN_00401010.disassembled.txt",
                "Function: FUN_00401010\nRET\n",
            )
            write_fixture(
                root,
                "analysis/FUN_00401010.decompiled.txt",
                "void FUN_00401010(void) {}\n",
            )
            write_fixture(
                root,
                "analysis/FUN_00401020.disassembled.txt",
                "Function: FUN_00401020\nRET\n",
            )
            write_fixture(
                root,
                "analysis/FUN_00401100.disassembled.txt",
                "Function: _malloc\nRET\n",
            )
            write_fixture(
                root,
                "analysis/FUN_00401100.decompiled.txt",
                "void * _malloc(void) {}\n",
            )

            self.assertEqual(
                ProjectRepository(root).next_unreconstructed_address(),
                0x00401010,
            )

    def test_next_target_skips_an_unsafe_existing_reconstruction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            write_fixture(
                root,
                "src/sample.c",
                """/* Function start: 0x401000 */
int ExistingFunction(void)
{
    return *(int *)0x00402040;
}
""",
            )
            write_fixture(
                root,
                "analysis/FUN_00401010.disassembled.txt",
                "Function: FUN_00401010\nRET\n",
            )
            write_fixture(
                root,
                "analysis/FUN_00401010.decompiled.txt",
                "void FUN_00401010(void) {}\n",
            )

            self.assertEqual(
                ProjectRepository(root).next_unreconstructed_address(),
                0x00401010,
            )

    def test_next_target_skips_configured_deferred_addresses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            config_path = root / "binary-recons.toml"
            config = config_path.read_text(encoding="utf-8")
            config = config.replace(
                "\n[[support_files]]",
                "\nskip_addresses = [0x00401010]\n\n[[support_files]]",
                1,
            )
            config_path.write_text(config, encoding="utf-8")
            write_fixture(
                root,
                "src/sample.c",
                "/* Function start: 0x401000 */\n"
                "int ExistingFunction(void)\n"
                "{\n"
                "    return 0;\n"
                "}\n",
            )
            for address in (0x00401010, 0x00401020):
                write_fixture(
                    root,
                    "analysis/FUN_%08X.disassembled.txt" % address,
                    "Function: FUN_%08X\nRET\n" % address,
                )
                write_fixture(
                    root,
                    "analysis/FUN_%08X.decompiled.txt" % address,
                    "void FUN_%08X(void) {}\n" % address,
                )

            repository = ProjectRepository(root)
            self.assertEqual(
                repository.next_unreconstructed_address(),
                0x00401020,
            )
            self.assertEqual(repository.resolve_target(0x00401010).address, 0x00401010)

    def test_next_target_requires_an_explicit_safe_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            config = (root / "binary-recons.toml").read_text(encoding="utf-8")
            config = config.split("\n[[source_units]]", 1)[0] + "\n"
            (root / "binary-recons.toml").write_text(config, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "exclude CRT and library code"):
                ProjectRepository(root).next_unreconstructed_address()

    def test_next_target_stops_before_exports_beyond_the_safe_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            write_fixture(
                root,
                "src/sample.c",
                "/* Function start: 0x401000 */\n"
                "int ExistingFunction(void)\n"
                "{\n"
                "    return 0;\n"
                "}\n",
            )
            write_fixture(
                root,
                "analysis/FUN_00401100.disassembled.txt",
                "Function: _malloc\nRET\n",
            )
            write_fixture(
                root,
                "analysis/FUN_00401100.decompiled.txt",
                "void * _malloc(void) {}\n",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "no unreconstructed function exports remain",
            ):
                ProjectRepository(root).next_unreconstructed_address()

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

    def test_address_only_cli_improves_an_existing_definition(self) -> None:
        class FakeClient:
            similarity_prompts: list[str] = []

            def __init__(self, *args: object):
                pass

            def infer_contract(self, *args: object):
                raise AssertionError("an existing definition keeps its contract")

            def propose_symbols(self, *args: object):
                raise AssertionError("an existing definition keeps its name")

            def repair_compile(self, *args: object):
                raise AssertionError("the existing definition compiles")

            def improve_similarity(self, prompt: str, round_number: int):
                type(self).similarity_prompts.append(prompt)
                if round_number != 1:
                    raise AssertionError("only one edit round was configured")
                return (
                    SimilarityPatch(edit=ExactEdit(old="return 7;", new="return 8;")),
                    {},
                )

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root, "return 8;")
            write_fixture(
                root,
                "analysis/FUN_00401000.disassembled.txt",
                "Function: FUN_00401000\nAddress: 0x00401000\n\nMOV EAX,0x8\nRET\n",
            )
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
            write_fixture(
                root,
                "include/functions.h",
                "int ReadFixtureValue(void); /* 0x00401000 */\n"
                "void ExistingFixtureAction(void); /* 0x00402000 */\n",
            )

            def compare(
                repository_self: ProjectRepository,
                target: object,
                timeout: object,
            ):
                source = (root / "src/sample.c").read_text(encoding="utf-8")
                score = 90.0 if "return 8;" in source else 70.0
                return score, (
                    "Comparison for function ReadFixtureValue\n"
                    "0x401000: MOV EAX,0x7 | 0x501000: MOV EAX,0x8\n"
                    "Similarity: %.2f%%" % score
                )

            with (
                patch.object(ProjectRepository, "compare", compare),
                patch("binary_recons.search.ManagedLlamaServer", FakeManagedServer),
                patch("binary_recons.search.StructuredModelClient", FakeClient),
                patch("binary_recons.cli.DEFAULT_MODEL_PATH", None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = main(
                    [
                        "--project-root",
                        str(root),
                        "--address",
                        "0x401000",
                        "--max-edits",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(len(FakeClient.similarity_prompts), 1)
            prompt = FakeClient.similarity_prompts[0]
            self.assertIn("DIFF ORIENTATION", prompt)
            self.assertIn("ORIGINAL FUNCTION ASSEMBLY", prompt)
            self.assertIn("MOV EAX,0x8", prompt)
            self.assertIn("GHIDRA DECOMPILATION", prompt)
            self.assertIn("return 8;", prompt)
            self.assertIn("CURRENT SOURCE", prompt)
            self.assertIn("return 7;", prompt)
            source = (root / "src/sample.c").read_text(encoding="utf-8")
            self.assertIn("int ReadFixtureValue(void)", source)
            self.assertIn("return 8;", source)
            self.assertFalse(
                list(
                    (root / "artifacts/reconstruction/local-model/00401000").glob(
                        "*/contract.prompt.txt"
                    )
                )
            )

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

    def test_new_support_globals_must_be_used_and_can_require_an_address(self) -> None:
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
    return g_fixtureValue;
}""",
                supporting_insertions=[
                    SupportingInsertion(
                        path="include/globals.h",
                        content=(
                            "extern int g_fixtureValue;\n"
                            "extern int g_unusedValue_00402044;"
                        ),
                    ),
                    SupportingInsertion(
                        path="src/globals.c",
                        content="int g_fixtureValue;\nint g_unusedValue_00402044;",
                    ),
                ],
            )
            with self.assertRaises(ValueError) as raised:
                validate_candidate(
                    candidate,
                    target.with_candidate_contract(candidate),
                    set(),
                    repository.allowed_support_paths(),
                    require_global_address_suffix=True,
                )
            self.assertIn("unused global", str(raised.exception))
            self.assertIn("hexadecimal address suffix", str(raised.exception))

    def test_feedback_distinguishes_compiler_errors_and_compacts_assembly(self) -> None:
        compiler = (
            "sample.c(3) : warning C4013: 'other' undefined\n"
            "sample.c(4) : error C2065: 'missing' : undeclared identifier"
        )
        comparison = """Comparison for function ReadFixtureValue
0x401000: MOV EAX,0x500000 | 0x500000: MOV EAX,0x600000
0x401005: ADD EAX,1 | 0x500005: SUB EAX,1
Similarity: 70.00%
"""
        self.assertTrue(ProjectRepository.has_compiler_errors(compiler))
        compile_feedback = ProjectRepository.compact_feedback(compiler, None)
        self.assertTrue(compile_feedback.startswith("sample.c(4) : error C2065"))
        self.assertNotIn("warning C4013", compile_feedback)
        rendered = "one\ntwo\nwarning source\nreturn missing;\nfive\n"
        focused = ProjectRepository.compact_compile_feedback(compiler, rendered)
        self.assertIn("FIRST BLOCKING SOURCE FILE LINE", focused)
        self.assertIn("return missing;", focused)
        self.assertIn("3: warning source", focused)
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
                        "bool CalculateFixtureTotal(int count, tagRECT *bounds, "
                        "long unused_flags)"
                    ),
                ),
                repository,
            )
            source = """undefined4 FUN_00401000(int param_1, tagRECT *param_2)
{
    bool result;
    long long accumulator;
    ExistingFixtureAction();
    puts(s_fixture_text_00403000);
    puts(&DAT_00403000);
    result = DAT_00402010[param_1] + DAT_00402030 * 4 + DAT_00409999;
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
                "int CalculateFixtureTotal(int count, RECT *bounds, long unused_flags)",
            )
            self.assertIn("g_fixture_value_00402010[count]", candidate.source)
            self.assertIn("DAT_00402030 * 4", candidate.source)
            self.assertNotIn("g_fixture_handle_00402030", candidate.source)
            self.assertIn("DAT_00409999", candidate.source)
            self.assertNotIn("(int *)0x", candidate.source)
            self.assertEqual(candidate.source.count('puts("fixture text");'), 2)
            self.assertNotIn("s_fixture_text_00403000", candidate.source)
            self.assertIn("int result;", candidate.source)
            self.assertIn("__int64 accumulator;", candidate.source)
            self.assertNotIn("long long", candidate.source)
            self.assertNotIn("param_1", candidate.source)
            self.assertTrue(
                any("parameter param_1 -> count" in item for item in changes)
            )
            self.assertTrue(
                any("mandatory source-level global repair" in item for item in changes)
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

    def test_absolute_pointer_and_unresolved_global_are_source_safety_errors(
        self,
    ) -> None:
        candidate = Candidate(
            symbol="CheckFixtureScore",
            prototype="void CheckFixtureScore(void)",
            source="""/* Function start: 0x401000 */
void CheckFixtureScore(void)
{
    int *raw_pointer = 0x00414980;
    LPVOID raw_buffer = (LPVOID)0x00416688;

    if (*(int *)0x004167a4 <= DAT_0041560c) {
        *(short *)((char *)0x004166a6 + 0x1c) = 1;
    }
}""",
        )
        violations = source_safety_violations(candidate)
        self.assertTrue(any("0x004167a4" in item for item in violations))
        self.assertTrue(any("0x004166a6" in item for item in violations))
        self.assertTrue(any("0x00414980" in item for item in violations))
        self.assertTrue(any("0x00416688" in item for item in violations))
        self.assertTrue(any("DAT_0041560c" in item for item in violations))
        underscored = candidate.model_copy(
            update={"source": candidate.source.replace("DAT_0041560c", "_DAT_0041560c")}
        )
        self.assertTrue(
            any(
                "_DAT_0041560c" in item
                for item in source_safety_violations(underscored)
            )
        )
        feedback = source_safety_feedback(candidate)
        self.assertIsNotNone(feedback)
        self.assertIn("SOURCE SAFETY ERROR:", feedback or "")

        repaired = candidate.model_copy(
            update={
                "source": """/* Function start: 0x401000 */
void CheckFixtureScore(void)
{
    if (g_fixture_scores_00416690[9].score <= g_fixture_score_0041560c) {
        g_fixture_scores_00416690[1].level = 1;
    }
}"""
            }
        )
        self.assertEqual(source_safety_violations(repaired), [])
        comment_only = repaired.model_copy(
            update={
                "source": repaired.source.replace(
                    "{\n",
                    "{\n    /* never use *(int *)0x004167a4 */\n"
                    '    const char *message = "(int *)0x004166a6";\n',
                    1,
                )
            }
        )
        self.assertEqual(source_safety_violations(comment_only), [])

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

    def test_sanitizer_rejects_a_new_absolute_pointer(self) -> None:
        source = """/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return g_fixture_value_00402010[0];
}"""
        raw = SourcePatch(
            edits=[
                ExactEdit(
                    old="g_fixture_value_00402010[0]",
                    new="*(int *)0x00402010",
                )
            ]
        )
        sanitized, rejected = sanitize_source_patch(
            source,
            raw,
            "ReadFixtureValue",
        )
        self.assertEqual(sanitized.edits, [])
        self.assertTrue(any("absolute-address pointer" in item for item in rejected))

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

    def test_patch_application_accumulates_configured_global_insertions(self) -> None:
        candidate = Candidate(
            symbol="ReadFixtureValue",
            prototype="int ReadFixtureValue(void)",
            source="""/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return DAT_00402040;
}""",
        )
        repaired = apply_source_patch(
            candidate,
            SourcePatch(
                identifier_replacements=[
                    IdentifierReplacement(
                        old="DAT_00402040",
                        new="g_fixture_score_00402040",
                    )
                ],
                supporting_insertions=[
                    SupportingInsertion(
                        path="include/globals.h",
                        content="extern int g_fixture_score_00402040;",
                    ),
                    SupportingInsertion(
                        path="src/globals.c",
                        content="int g_fixture_score_00402040;",
                    ),
                ],
            ),
            0x00401000,
        )
        self.assertIn("g_fixture_score_00402040", repaired.source)
        self.assertEqual(len(repaired.supporting_insertions), 2)

    def test_paired_support_declarations_bind_address_backed_array_uses(self) -> None:
        source = """/* Function start: 0x401000 */
HGDIOBJ ReadFixtureFrame(int frameIndex)
{
    DeleteObject(DAT_00402040);
    DeleteObject(_DAT_00402044);
    return *(HGDIOBJ *)(&DAT_00402040 + frameIndex * 4);
}"""
        insertions = [
            SupportingInsertion(
                path="include/globals.h",
                content="extern HGDIOBJ g_fixtureFrames_00402040[2];",
            ),
            SupportingInsertion(
                path="src/globals.c",
                content="HGDIOBJ g_fixtureFrames_00402040[2];",
            ),
        ]

        bound, changes = bind_supporting_address_symbols(source, insertions)

        self.assertIn("DeleteObject(g_fixtureFrames_00402040[0]);", bound)
        self.assertIn("DeleteObject(g_fixtureFrames_00402040[1]);", bound)
        self.assertIn("return g_fixtureFrames_00402040[frameIndex];", bound)
        self.assertNotIn("DAT_", bound)
        self.assertTrue(changes)

    def test_address_binding_requires_a_paired_declaration(self) -> None:
        source = "return DAT_00402040;"
        bound, changes = bind_supporting_address_symbols(
            source,
            [
                SupportingInsertion(
                    path="include/globals.h",
                    content="extern int g_fixtureValue_00402040;",
                )
            ],
        )
        self.assertEqual(bound, source)
        self.assertEqual(changes, [])


class StagedSearchTests(unittest.TestCase):
    def test_existing_absolute_pointer_must_be_repaired_before_comparison(
        self,
    ) -> None:
        class FakeClient:
            prompts: list[str] = []

            def __init__(self, *args: object):
                pass

            def repair_compile(self, prompt: str, round_number: int):
                type(self).prompts.append(prompt)
                if round_number != 1:
                    raise AssertionError("unexpected safety-repair round")
                if "SOURCE SAFETY ERROR:" not in prompt:
                    raise AssertionError("source-safety feedback was not supplied")
                if "CONFIGURED SUPPORT FILES" not in prompt:
                    raise AssertionError("support-file context was not supplied")
                return (
                    SourcePatch(
                        edits=[
                            ExactEdit(
                                old="*(int *)0x00402040",
                                new="g_fixture_score_00402040",
                            )
                        ],
                        supporting_insertions=[
                            SupportingInsertion(
                                path="include/globals.h",
                                content="extern int g_fixture_score_00402040;",
                            ),
                            SupportingInsertion(
                                path="src/globals.c",
                                content="int g_fixture_score_00402040;",
                            ),
                        ],
                    ),
                    {},
                )

            def improve_similarity(self, *args: object):
                raise AssertionError("the safety repair reaches the target")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            write_fixture(
                root,
                "src/sample.c",
                """#include "project.h"

/* Function start: 0x401000 */
int ReadFixtureValue(void)
{
    return *(int *)0x00402040;
}
""",
            )
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)
            compare_calls = 0

            def compare(*args: object):
                nonlocal compare_calls
                compare_calls += 1
                source = target.source_path.read_text(encoding="utf-8")
                self.assertNotIn("(int *)0x", source)
                self.assertIn("g_fixture_score_00402040", source)
                self.assertIn(
                    "extern int g_fixture_score_00402040;",
                    (root / "include/globals.h").read_text(encoding="utf-8"),
                )
                self.assertIn(
                    "int g_fixture_score_00402040;",
                    (root / "src/globals.c").read_text(encoding="utf-8"),
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
                    SearchConfig(seed=target.address, max_edits=1),
                    LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN),
                ).run()

            self.assertTrue(result.target_reached)
            self.assertEqual(compare_calls, 1)
            self.assertEqual(len(FakeClient.prompts), 1)
            self.assertNotIn(
                "(int *)0x",
                target.source_path.read_text(encoding="utf-8"),
            )

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
            self.assertEqual(manifest["workflow"], "ghidra-seed-bounded-qwen-edits-v2")
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
            self.assertTrue(edit_result["followed"])

    def test_compile_repair_follows_a_linker_failure_to_the_next_fix(self) -> None:
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
                type(self).compile_prompts.append(prompt)
                if round_number == 1:
                    return (
                        SourcePatch(
                            edits=[
                                ExactEdit(
                                    old="unknown_fixture_value",
                                    new="7",
                                    mode="all",
                                )
                            ]
                        ),
                        {},
                    )
                if round_number == 2:
                    if "LNK2001" not in prompt or "LNK1120" not in prompt:
                        raise AssertionError("linker failure was not followed")
                    return (
                        SourcePatch(edits=[ExactEdit(old="_rand()", new="rand()")]),
                        {},
                    )
                raise AssertionError("unexpected compile-repair round")

            def improve_similarity(self, *args: object):
                raise AssertionError("the second repair reaches the target")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(
                root,
                "return unknown_fixture_value + _rand();",
            )
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)

            def compare(*args: object):
                source = target.source_path.read_text(encoding="utf-8")
                if "unknown_fixture_value" in source:
                    return None, (
                        "sample.c(4) : error C2065: 'unknown_fixture_value' : "
                        "undeclared identifier"
                    )
                if "_rand()" in source:
                    return None, (
                        "sample.obj : error LNK2001: unresolved external symbol "
                        "__rand\n"
                        "sample.exe : fatal error LNK1120: 1 unresolved externals"
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
            self.assertEqual(result.score, 84.0)
            self.assertEqual(len(FakeClient.compile_prompts), 2)
            first_result = json.loads(
                (result.session_directory / "edit-01.compile.result.json").read_text()
            )
            self.assertTrue(first_result["followed"])
            self.assertIsNone(first_result["score"])

    def test_search_can_follow_a_regression_back_to_an_earlier_candidate(
        self,
    ) -> None:
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
                if round_number == 1:
                    if "Similarity: 70.00%" not in prompt:
                        raise AssertionError("baseline feedback was not supplied")
                    edit = ExactEdit(old="return 7;", new="return 8;")
                elif round_number == 2:
                    if "Similarity: 60.00%" not in prompt or "return 8;" not in prompt:
                        raise AssertionError("regressed trajectory was not followed")
                    edit = ExactEdit(old="return 8;", new="return 7;")
                else:
                    raise AssertionError("unexpected similarity round")
                return (
                    SimilarityPatch(edit=edit),
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
                    SearchConfig(seed=target.address, max_edits=2),
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
            self.assertTrue(decision["followed"])
            self.assertIn("return 8;", decision["candidate"]["source"])
            return_result = json.loads(
                (
                    result.session_directory / "edit-02.similarity.result.json"
                ).read_text()
            )
            self.assertTrue(return_result["followed"])
            self.assertIn("return 7;", return_result["candidate"]["source"])

    def test_search_follows_a_temporary_build_failure_to_a_better_result(
        self,
    ) -> None:
        class FakeClient:
            similarity_prompts: list[str] = []
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
                type(self).compile_prompts.append(prompt)
                if round_number != 2:
                    raise AssertionError("only the temporary failure needs repair")
                if "return 8;" not in prompt or "error C2065" not in prompt:
                    raise AssertionError("repair did not follow the working trajectory")
                return (
                    SourcePatch(edits=[ExactEdit(old="return 8;", new="return 9;")]),
                    {},
                )

            def improve_similarity(self, prompt: str, round_number: int):
                type(self).similarity_prompts.append(prompt)
                if round_number == 1:
                    return (
                        SimilarityPatch(
                            edit=ExactEdit(old="return 7;", new="return 8;")
                        ),
                        {},
                    )
                if round_number != 3:
                    raise AssertionError("unexpected similarity round")
                if "return 9;" not in prompt or "Similarity: 65.00%" not in prompt:
                    raise AssertionError(
                        "similarity edit did not follow the repaired source"
                    )
                return (
                    SimilarityPatch(edit=ExactEdit(old="return 9;", new="return 10;")),
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
                if "return 8;" in source:
                    return None, (
                        "sample.c(4) : error C2065: 'synthetic' : undeclared identifier"
                    )
                if "return 10;" in source:
                    score = 85.0
                elif "return 9;" in source:
                    score = 65.0
                else:
                    score = 70.0
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
                    SearchConfig(seed=target.address, max_edits=3),
                    LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN),
                ).run()

            self.assertTrue(result.target_reached)
            self.assertEqual(result.score, 85.0)
            self.assertEqual(len(FakeClient.similarity_prompts), 2)
            self.assertEqual(len(FakeClient.compile_prompts), 1)
            source = target.source_path.read_text(encoding="utf-8")
            self.assertIn("return 10;", source)
            self.assertNotIn("return 8;", source)
            self.assertNotIn("return 9;", source)

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
    def test_incomplete_structured_output_becomes_a_model_request_error(self) -> None:
        class FakeCompletions:
            def create_with_completion(self, **kwargs: object):
                raise IncompleteOutputException()

        class FakeInstructor:
            class Chat:
                completions = FakeCompletions()

            chat = Chat()

        class FakeOpenAI:
            def close(self) -> None:
                pass

        class FakeServer:
            base_url = "http://127.0.0.1:8080/v1"
            config = LlamaServerConfig(model_path=None, preset=ModelPreset.QWEN)

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
                SearchConfig(seed=1),
            )
            with self.assertRaisesRegex(
                ModelRequestError,
                "similarity edit request failed",
            ):
                client.improve_similarity("edit prompt", 1)
            client.close()

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
                if response_model is SimilarityPatch:
                    return (
                        SimilarityPatch(
                            edit=ExactEdit(old="return 7;", new="return 8;")
                        ),
                        {},
                    )
                if response_model is SourcePatch:
                    return (
                        SourcePatch(edits=[ExactEdit(old="missing", new="declared")]),
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
                SearchConfig(seed=1),
            )
            client.infer_contract("contract prompt")
            client.propose_symbols("name prompt", 1)
            client.repair_compile("compile prompt", 1)
            client.repair_compile("retry compile prompt", 2)
            client.improve_similarity("first edit prompt", 1)
            client.improve_similarity("retry edit prompt", 2)
            client.close()

        (
            contract_call,
            naming_call,
            compile_call,
            retry_compile_call,
            first_edit_call,
            retry_edit_call,
        ) = calls
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
        self.assertEqual(compile_call["max_tokens"], 768)
        self.assertEqual(compile_call["temperature"], 0.2)
        self.assertEqual(compile_call["presence_penalty"], 0.0)
        self.assertEqual(retry_compile_call["max_tokens"], 768)
        self.assertEqual(retry_compile_call["temperature"], 0.7)
        self.assertEqual(retry_compile_call["presence_penalty"], 1.5)
        self.assertEqual(first_edit_call["max_tokens"], 512)
        self.assertEqual(first_edit_call["temperature"], 0.2)
        self.assertEqual(first_edit_call["presence_penalty"], 0.0)
        self.assertEqual(retry_edit_call["max_tokens"], 512)
        self.assertEqual(retry_edit_call["temperature"], 0.7)
        self.assertEqual(retry_edit_call["presence_penalty"], 1.5)


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

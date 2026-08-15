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

from binary_recons.cli import main  # noqa: E402
from binary_recons.llama_server import ManagedLlamaServer  # noqa: E402
from binary_recons.models import (  # noqa: E402
    Candidate,
    CandidateBatch,
    LlamaServerConfig,
    ModelPreset,
    SearchConfig,
    ServerMode,
)
from binary_recons.repository import (  # noqa: E402
    ProjectRepository,
    current_function,
    replace_or_insert_function,
)
from binary_recons.search import ReconstructionSearch  # noqa: E402


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
        'rule_profiles = ["c89"]\n'
        'prompt_files = ["RECONSTRUCTION.md"]\n'
        'compare_command = ["fixture-compare", "{symbol}", "{address_hex}"]\n'
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
        "int sample_function(void); /* 0x00401000 */\n",
    )
    write_fixture(
        root,
        "include/globals.h",
        "extern int g_sample_value_00402010[4];\n"
        "extern SampleVector g_sample_vector_00402020[4];\n",
    )
    write_fixture(
        root,
        "analysis/FUN_00401000.disassembled.txt",
        "Function: sample_function\n"
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
    def test_target_is_inferred_from_explicit_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            target = ProjectRepository(root).resolve_target(0x00401000)
            self.assertEqual(target.symbol, "sample_function")
            self.assertEqual(target.source_path, (root / "src/sample.c").resolve())
            self.assertEqual(target.prototype, "int sample_function(void)")

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
                repository.config.comparison_command(target.symbol, target.address),
                ["fixture-compare", "sample_function", "00401000"],
            )

    def test_candidate_schema_accepts_large_functions(self) -> None:
        candidate = Candidate(source="x" * 6000)
        self.assertEqual(len(candidate.source), 6000)

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


class CompileRepairTests(unittest.TestCase):
    def test_compiler_failure_gets_one_focused_model_repair(self) -> None:
        failing_candidate = """/* Function start: 0x401000 */
int sample_function(void)
{
    return missing_value;
}"""
        repaired_candidate = """/* Function start: 0x401000 */
int sample_function(void)
{
    return 7;
}"""

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
                    CandidateBatch(candidates=[Candidate(source=failing_candidate)]),
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
                return Candidate(source=repaired_candidate), {"kind": "repair"}

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture_project(root)
            repository = ProjectRepository(root)
            target = repository.resolve_target(0x00401000)

            def compare_candidate(*args: object) -> tuple[float | None, str]:
                source = target.source_path.read_text(encoding="utf-8")
                if "missing_value" in source:
                    return (
                        None,
                        "sample.c(4) : error C2065: 'missing_value' : "
                        "undeclared identifier",
                    )
                if "return 7;" in source:
                    return 100.0, "Similarity: 100.00%"
                raise AssertionError("unexpected candidate source")

            repository.compare = compare_candidate  # type: ignore[method-assign]
            config = SearchConfig(
                seed=1,
                max_iterations=1,
                candidates_per_iteration=1,
                compile_repair_attempts=1,
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
            self.assertEqual(result.attempts, 2)
            self.assertEqual(FakeGenerator.repair_calls, 1)
            self.assertIn("missing_value", FakeGenerator.repair_prompts[0])
            self.assertIn("error C2065", FakeGenerator.repair_prompts[0])
            self.assertIn("return 7;", target.source_path.read_text(encoding="utf-8"))
            repair_logs = list(result.session_directory.glob("*-repair-01.c"))
            self.assertEqual(len(repair_logs), 1)


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
            model_path=Path("qwen.gguf"),
            preset=ModelPreset.QWEN,
        )
        gemma = LlamaServerConfig(
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

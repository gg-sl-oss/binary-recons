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
    LlamaServerConfig,
    SearchConfig,
    ServerMode,
)
from binary_recons.repository import (  # noqa: E402
    ProjectRepository,
    current_function,
    replace_or_insert_function,
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
        "docs/ORDER.md",
        "| `src/sample.c` | `0x401000`–`0x4010FF` | fixture |\n",
    )
    write_fixture(root, "src/sample.c", '#include "project.h"\n')
    write_fixture(
        root,
        "include/wc1funcs.h",
        "int sample_function(void); /* 0x00401000 */\n",
    )
    write_fixture(root, "include/wc1extern.h", "")
    write_fixture(
        root,
        "include/globals.h",
        "extern int g_sample_value_00402010[4];\n"
        "extern SampleVector g_sample_vector_00402020[4];\n",
    )
    write_fixture(
        root,
        "code-full/FUN_00401000.disassembled.txt",
        "Function: sample_function\n"
        "Address: 0x00401000\n\n"
        "MOV EAX,0x7\n"
        "CMP EAX,0x7\n"
        "RET\n",
    )
    write_fixture(
        root,
        "code-full/FUN_00401000.decompiled.txt",
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
                (root / "out/qwen-reconstruct/00401000").glob(
                    "*/iteration-01.prompt.txt"
                )
            )
            self.assertEqual(len(prompts), 1)

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
            hint = ProjectRepository(Path(temporary))._concise_decompilation(
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

    def test_candidate_schema_accepts_large_functions(self) -> None:
        candidate = Candidate(source="x" * 6000)
        self.assertEqual(len(candidate.source), 6000)


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


if __name__ == "__main__":
    unittest.main()

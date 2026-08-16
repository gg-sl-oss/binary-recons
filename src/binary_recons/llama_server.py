"""Owned lifecycle for the standalone llama.cpp HTTP server."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from types import TracebackType

import httpx

from .models import LlamaServerConfig, SearchConfig, ServerMode
from .utils import tail_text, utc_now, write_json


class ManagedLlamaServer:
    def __init__(
        self,
        config: LlamaServerConfig,
        search: SearchConfig,
        run_directory: Path,
    ):
        self.config = config
        self.search = search
        self.run_directory = run_directory
        self.log_path = run_directory / "llama-server.log"
        self.manifest_path = run_directory / "llama-server.json"
        self.process: subprocess.Popen[str] | None = None
        self._log_handle = None
        self._owned = False
        self._manifest: dict[str, object] = {
            "mode": config.mode.value,
            "health_url": config.health_url,
            "base_url": config.base_url,
            "status": "created",
        }

    def __enter__(self) -> "ManagedLlamaServer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop(reason="exception" if exc is not None else "normal")

    @property
    def base_url(self) -> str:
        return self.config.base_url

    def _write_manifest(self) -> None:
        write_json(self.manifest_path, self._manifest)

    def _health_status(self) -> tuple[int | None, str]:
        try:
            response = httpx.get(self.config.health_url, timeout=2.0)
        except httpx.HTTPError as error:
            return None, str(error)
        return response.status_code, response.text[:1000]

    def start(self) -> None:
        status, body = self._health_status()
        if self.config.mode == ServerMode.EXTERNAL:
            if status != 200:
                raise RuntimeError(
                    "external llama-server is not ready at %s: HTTP %s %s"
                    % (self.config.health_url, status, body)
                )
            self._manifest.update(
                {
                    "status": "ready",
                    "owned": False,
                    "ready_at": utc_now(),
                }
            )
            self._write_manifest()
            return

        if status is not None:
            raise RuntimeError(
                "a server already answers at %s (HTTP %s); use "
                "--server-mode external to reuse it" % (self.config.health_url, status)
            )
        if self.config.command_override is None:
            binary = self.config.resolved_binary()
            if not binary.exists():
                raise RuntimeError("llama-server binary does not exist: %s" % binary)
            if self.config.model_path is None:
                raise RuntimeError(
                    "no model configured; pass --model-path, set "
                    "BINARY_RECONS_MODEL_PATH, or cache a Qwen GGUF under "
                    "~/.cache/huggingface/hub"
                )
            if not self.config.model_path.exists():
                raise RuntimeError("model does not exist: %s" % self.config.model_path)

        command = self.config.command(self.search)
        self._log_handle = self.log_path.open("w", encoding="utf-8", buffering=1)
        started = time.monotonic()
        self.process = subprocess.Popen(
            command,
            cwd=self.run_directory,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self._owned = True
        self._manifest.update(
            {
                "status": "loading",
                "owned": True,
                "pid": self.process.pid,
                "command": command,
                "started_at": utc_now(),
            }
        )
        self._write_manifest()

        deadline = started + self.config.startup_timeout
        try:
            while time.monotonic() < deadline:
                self.ensure_alive(check_health=False)
                status, body = self._health_status()
                if status == 200:
                    self._manifest.update(
                        {
                            "status": "ready",
                            "ready_at": utc_now(),
                            "startup_seconds": round(time.monotonic() - started, 3),
                        }
                    )
                    self._write_manifest()
                    return
                if status not in (None, 503):
                    raise RuntimeError(
                        "unexpected llama-server health response: HTTP %s %s"
                        % (status, body)
                    )
                time.sleep(self.config.health_interval)
            raise RuntimeError(
                "llama-server did not become ready within %.1f seconds"
                % self.config.startup_timeout
            )
        except BaseException:
            self.stop(reason="startup-failure")
            raise

    def ensure_alive(self, check_health: bool = True) -> None:
        if self._owned and self.process is not None:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    "llama-server exited unexpectedly with status %d\n%s"
                    % (return_code, tail_text(self.log_path))
                )
        if check_health:
            status, body = self._health_status()
            if status != 200:
                raise RuntimeError(
                    "llama-server is not healthy: HTTP %s %s\n%s"
                    % (status, body, tail_text(self.log_path))
                )

    def stop(self, reason: str = "normal") -> None:
        if self._owned and self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=self.config.shutdown_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)

        return_code = self.process.poll() if self.process is not None else None
        self._manifest.update(
            {
                "status": "stopped" if self._owned else "released",
                "stopped_at": utc_now(),
                "stop_reason": reason,
                "return_code": return_code,
            }
        )
        self._write_manifest()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

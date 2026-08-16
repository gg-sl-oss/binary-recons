"""Command-line interface for bounded local-model reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model_client import ModelRequestError
from .models import (
    Candidate,
    DEFAULT_MODEL_PATH,
    LlamaServerConfig,
    ModelPreset,
    SearchConfig,
    ServerMode,
)
from .repository import ProjectRepository
from .search import ReconstructionSearch


def _integer(value: str) -> int:
    return int(value, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate a Ghidra C seed with bounded Instructor/Qwen exact edits"
        )
    )
    target = parser.add_argument_group("target")
    target.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="target repository root (default: current directory)",
    )
    target.add_argument(
        "--config",
        type=Path,
        help="project configuration (default: <project-root>/binary-recons.toml)",
    )
    selection = target.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--address",
        type=_integer,
        help="target a specific function address",
    )
    selection.add_argument(
        "--next-function",
        action="store_true",
        help=(
            "select the next unreconstructed export inside the configured "
            "source_units safety ranges"
        ),
    )
    target.add_argument(
        "--symbol",
        help="optional fixed symbol; default: model proposes one for new functions",
    )
    target.add_argument(
        "--source",
        type=Path,
        help="default: infer from an existing marker or configured source range",
    )
    target.add_argument(
        "--prototype",
        help="optional fixed prototype; default: model proposes one for new functions",
    )
    target.add_argument(
        "--reopen-contract",
        action="store_true",
        help=(
            "discard an existing inferred name/prototype for this run and ask the "
            "model to infer a replacement"
        ),
    )
    target.add_argument(
        "--resume-candidate",
        type=Path,
        help=(
            "seed from a logged candidate JSON or selected-change-set.json "
            "without requesting a new contract"
        ),
    )

    search = parser.add_argument_group("search")
    search.add_argument(
        "--max-edits",
        "--max-iterations",
        dest="max_edits",
        type=int,
        default=4,
        help="maximum one-edit Qwen rounds after the mechanical seed (default: 4)",
    )
    search.add_argument("--target-score", type=float, default=80.0)
    search.add_argument(
        "--max-tokens",
        type=int,
        default=768,
        help="hard output-token cap; each staged request applies a lower cap",
    )
    search.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium"),
        default="none",
    )
    search.add_argument(
        "--max-callees",
        type=int,
        default=2,
        help="direct callee exports to collect (default: 2; raise only when needed)",
    )
    search.add_argument("--request-timeout", type=float, default=60.0)
    search.add_argument("--build-timeout", type=float, default=120.0)
    search.add_argument("--format-retries", type=int, default=1)
    search.add_argument(
        "--seed",
        type=_integer,
        help="default: target function address",
    )
    search.add_argument("--temperature", type=float)
    search.add_argument("--top-p", type=float)
    search.add_argument("--top-k", type=int)
    search.add_argument("--min-p", type=float, default=0.0)
    search.add_argument("--presence-penalty", type=float)
    search.add_argument("--repeat-penalty", type=float, default=1.0)
    search.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="write the mechanically generated prompt without starting a model",
    )

    server = parser.add_argument_group("llama.cpp server")
    server.add_argument(
        "--server-mode",
        choices=tuple(mode.value for mode in ServerMode),
        default=ServerMode.MANAGED.value,
        help="managed starts/stops llama-server; external reuses an existing one",
    )
    server.add_argument("--llama-bin", type=Path)
    server.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "GGUF path (default: BINARY_RECONS_MODEL_PATH, then a cached Qwen "
            "model under ~/.cache/huggingface/hub)"
        ),
    )
    server.add_argument("--model", help="llama-server alias (default: GGUF filename)")
    server.add_argument(
        "--model-preset",
        choices=tuple(preset.value for preset in ModelPreset),
        default=ModelPreset.AUTO.value,
        help="model-specific serving/template defaults (default: infer from name)",
    )
    server.add_argument("--server-host", default="127.0.0.1")
    server.add_argument("--server-port", type=int, default=8080)
    server.add_argument("--ctx-size", type=int, default=32768)
    server.add_argument("--startup-timeout", type=float, default=240.0)
    server.add_argument("--shutdown-timeout", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reopen_contract and (args.symbol is not None or args.prototype is not None):
        parser.error(
            "--reopen-contract cannot be combined with --symbol or --prototype"
        )
    repository = ProjectRepository(args.project_root, args.config)
    address = (
        repository.next_unreconstructed_address()
        if args.next_function
        else args.address
    )
    if address is None:
        raise RuntimeError("internal error: no target selection was parsed")
    if args.next_function:
        print(
            "selected next unreconstructed target 0x%08X from source_units" % address,
            flush=True,
        )
    target = repository.resolve_target(
        address=address,
        symbol=args.symbol,
        source=args.source,
        prototype=args.prototype,
    )
    if args.reopen_contract:
        target = target.model_copy(update={"symbol": None, "prototype": None})
    model_path = args.model_path
    model_alias = args.model
    if model_alias is None:
        model_alias = model_path.stem if model_path is not None else "local-model"
    resume_candidate = None
    if args.resume_candidate is not None:
        payload = json.loads(args.resume_candidate.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "candidate" in payload:
            payload = payload["candidate"]
        resume_candidate = Candidate.model_validate(payload)
    search_config = SearchConfig(
        model=model_alias,
        max_edits=args.max_edits,
        target_score=args.target_score,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        max_callees=args.max_callees,
        request_timeout=args.request_timeout,
        build_timeout=args.build_timeout,
        format_retries=args.format_retries,
        seed=args.seed if args.seed is not None else target.address,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repeat_penalty=args.repeat_penalty,
    )
    server_config = LlamaServerConfig(
        mode=ServerMode(args.server_mode),
        binary=args.llama_bin,
        model_path=args.model_path,
        alias=model_alias,
        preset=ModelPreset(args.model_preset),
        host=args.server_host,
        port=args.server_port,
        context_size=args.ctx_size,
        startup_timeout=args.startup_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )
    print(
        "target 0x%08X: %s -> %s"
        % (
            target.address,
            target.symbol or "<model-inferred contract>",
            target.source_display,
        ),
        flush=True,
    )
    result = ReconstructionSearch(
        repository,
        target,
        search_config,
        server_config,
        initial_candidate=resume_candidate,
    ).run(dry_run_prompt=args.dry_run_prompt)
    return 0 if result.score is not None or args.dry_run_prompt else 1


def entrypoint() -> int:
    try:
        return main()
    except (
        RuntimeError,
        OSError,
        ValueError,
        ModelRequestError,
    ) as error:
        print("binary-recons: %s" % error, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("binary-recons: interrupted", file=sys.stderr)
        return 130

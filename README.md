# binary-recons

[![CI](https://github.com/gg-sl-oss/binary-recons/actions/workflows/ci.yml/badge.svg)](https://github.com/gg-sl-oss/binary-recons/actions/workflows/ci.yml)

`binary-recons` runs a bounded generate, compile, and assembly-compare loop for
recovering C source with a local Qwen model. Python controls every repository
edit and `binary-comp` comparison; the model receives structured evidence and
never receives shell or filesystem tools.

The first project adapter expects the Wing Commander reconstruction layout:
Ghidra exports in `code-full/`, declarations in `include/`, compilation-unit
ranges in `docs/ORDER.md`, and `make compare-func FUNC=<name>`.

## Install

Python 3.11+, `llama-server`, and the Qwen3.8 27B GGUF are required.

```sh
python3 -m pip install -e /path/to/binary-recons
```

The command discovers the Unsloth Hugging Face cache automatically. Override
the model when needed:

```sh
export BINARY_RECONS_MODEL_PATH=/path/to/Qwen3.8-27B-BF16-00001-of-00002.gguf
```

## Run

Run from a reconstruction repository; the current directory is the default
project root:

```sh
cd /path/to/wc1-test
binary-recons --address 0x418210
```

Or provide the repository explicitly:

```sh
binary-recons --project-root /path/to/wc1-test --address 0x418210
```

Managed mode starts one `llama-server`, waits for its health endpoint, records
its PID and logs, and stops only that owned process group on exit. To reuse a
server started elsewhere, opt into non-owning mode:

```sh
binary-recons --address 0x418210 --server-mode external
```

Use `--dry-run-prompt` to collect evidence and inspect the prompt without
loading Qwen. Runs are recorded under
`out/qwen-reconstruct/<address>/<timestamp>/` in the target repository.

The test suite and GitHub CI use a synthetic HTTP server and require neither a
local model nor `llama-server`.

# binary-recons

[![CI](https://github.com/gg-sl-oss/binary-recons/actions/workflows/ci.yml/badge.svg)](https://github.com/gg-sl-oss/binary-recons/actions/workflows/ci.yml)

`binary-recons` runs a bounded generate, compile, and assembly-compare loop for
recovering one source function at a time with a local language model. Python
owns repository edits and comparisons; the model receives a structured prompt
and never receives shell or filesystem tools.

Target-specific layout, compiler details, reconstruction rules, and comparison
commands belong in the target repository. The Python package contains no
conventions for a particular binary or source tree.

## Install

Python 3.11+ is required. Managed model serving also requires `llama-server`.

```sh
python3 -m pip install -e /path/to/binary-recons
```

The installed `binary-recons` command can be invoked from any directory. CI and
the unit tests use a synthetic HTTP server, so installing or downloading a
model is not required to test the package.

## Configure a target project

Create `binary-recons.toml` at the target repository root:

```toml
schema_version = 1
language = "C"
compiler = "Microsoft Visual C++ 4.20"
exports_dir = "ghidra"
output_dir = "out/binary-recons"
source_dirs = ["src"]
declaration_files = ["include/*.h"]
prompt_files = ["RECONSTRUCTION.md"]
compare_command = ["make", "compare-func", "FUNC={symbol}", "ADDR={address_hex}"]

[[source_units]]
path = "src/game.c"
start = 0x00401000
end = 0x0040ffff
```

The comparison command may use `{symbol}`, `{address}`, and `{address_hex}`.
It must compile the current source and print a `Similarity: N%` line from the
project's comparison authority.

Assembly and decompiler exports use these names by default:

```text
ghidra/FUN_00401000.disassembled.txt
ghidra/FUN_00401000.decompiled.txt
```

The assembly export should begin with `Function: <symbol>` when the source does
not yet contain a definition. A declaration matching that symbol must exist in
one of `declaration_files`, or `--prototype` must be supplied. Project-specific
rules live in the configured prompt files and are included verbatim in each
request.

## Run

Set a GGUF once, then run against any configured project:

```sh
export BINARY_RECONS_MODEL_PATH=/models/model.gguf
binary-recons --project-root /path/to/target --address 0x401000
```

`--model-preset auto` identifies Qwen and Gemma from the alias or filename.
Explicit presets are useful when the filename is ambiguous:

```sh
binary-recons --project-root /path/to/target --address 0x401000 \
  --model-path /models/gemma-4.gguf --model-preset gemma
```

The Gemma preset uses Gemma's recommended sampling defaults. The Qwen preset
adds llama.cpp's Qwen MTP speculative-decoding flags. CLI sampling options
override either preset.

Managed mode starts one `llama-server`, monitors it, records its PID and logs,
and stops only that owned process group on exit. To reuse an already-running
server without taking ownership of it:

```sh
binary-recons --project-root /path/to/target --address 0x401000 \
  --server-mode external
```

Use `--dry-run-prompt` to inspect all collected evidence without loading a
model. Runs are recorded under the configured `output_dir`, grouped by model,
address, and timestamp.

Each model response is validated before it can replace source. Candidates are
compiled and compared one at a time, and only the best result is retained. If a
candidate reaches the compiler but has compiler errors, a separately bounded
repair request receives the failing definition and diagnostics. Set
`--compile-repair-attempts 0` to disable that step.

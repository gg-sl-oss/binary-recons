# binary-recons

[![CI](https://github.com/gg-sl-oss/binary-recons/actions/workflows/ci.yml/badge.svg)](https://github.com/gg-sl-oss/binary-recons/actions/workflows/ci.yml)

`binary-recons` turns a Ghidra C decompilation into a compiling source function,
then makes a few measured source-form edits against the original assembly. It is
designed for fast local-model passes: Python owns the workflow and Qwen receives
one small, typed decision at a time.

The validated default model is
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
using the BF16 GGUF. Other llama.cpp models remain usable, but the prompts,
sampling, and edit sizes are tuned for Qwen.

## How the workflow works

For a new function, one run performs these stages:

1. Read the original assembly, Ghidra decompilation, referenced declarations,
   strings, callees, and project rules.
2. Ask Qwen only for a meaningful function name and complete C prototype.
3. Build the first candidate mechanically from Ghidra: align parameter names,
   normalize configured historical-C spellings, map address-backed declarations,
   and give unresolved `DAT_<address>` values a typed absolute-address fallback.
4. Apply the candidate as a transaction, compile it, and run the project's
   configured assembly comparison command.
5. If compilation fails, put the compact compiler diagnostics first in a Qwen
   request for at most one exact edit and eight whole-token replacements.
6. Once it compiles, give Qwen a compact normalized assembly diff together with
   the original function assembly, Ghidra semantic hint, and relevant existing
   declarations, then request one exact source-form edit.
7. Accept a trial only when it reduces the compiler-error count or strictly
   increases measured similarity. Otherwise roll it back and include the exact
   rejected patch plus its compiler or similarity result in the next turn. Stop
   at the target or after the bounded edit budget.

Existing functions keep their established name and prototype and start at stage
4. The model never receives shell access, unrestricted filesystem tools, or an
agent loop. Instructor constrains every response with a JSON schema, and Python
is the only component that writes, builds, compares, accepts, or rolls back.

The Ghidra export is the sole candidate seed. There is no alternate draft-source
input and no full-function generation batch in the normal path.

## Install

Python 3.11+ is required. Managed model serving also requires a recent
`llama-server` on `PATH`.

```sh
python3 -m pip install -e /path/to/binary-recons
```

The installed `binary-recons` command works from any directory. CI and the unit
tests use synthetic projects and servers, so they do not download or run Qwen.

## Configure a target project

Create `binary-recons.toml` at the target repository root:

```toml
schema_version = 1
language = "C"
compiler = "Microsoft Visual C++ 4.20"
exports_dir = "ghidra"
output_dir = "out/binary-recons"
strings_file = "ghidra/strings.txt"
source_dirs = ["src"]
declaration_files = ["include/*.h"]
prototype_file = "include/functions.h"
rule_profiles = ["c89", "msvc4-od"]
prompt_files = ["RECONSTRUCTION.md"]
compare_command = ["make", "compare-func", "FUNC={symbol}", "ADDR={address_hex}"]

[[source_units]]
path = "src/game.c"
start = 0x00401000
end = 0x0040ffff
```

The comparison command may use `{symbol}`, `{address}`, and `{address_hex}`. It
must compile the current source and print a `Similarity: N%` line from the
project's comparison authority.

Assembly and decompiler exports use these names by default:

```text
ghidra/FUN_00401000.disassembled.txt
ghidra/FUN_00401000.decompiled.txt
```

Reusable language/compiler rules belong in packaged `rule_profiles`; facts
unique to a target belong in `prompt_files`. The current shared profiles are
`c89` and `msvc4-od`.

`support_files` remain an optional transaction boundary for resuming an older
logged `Candidate` that already contains bounded supporting insertions. The
normal compile-first workflow does not ask Qwen to create header, global, helper,
or wrapper edits; it writes only the target definition and the driver-managed
prototype.

## Run with Qwen

Point the tool at the first shard of the BF16 GGUF once:

```sh
export BINARY_RECONS_MODEL_PATH=/models/Qwen3.8-27B-BF16-00001-of-00002.gguf
binary-recons --project-root /path/to/target --address 0x401000
```

That address-only command is intentionally identical for a new target and an
existing reconstruction. If the source unit already contains the address
marker, `binary-recons` discovers its definition and established prototype,
measures it as the baseline, and spends the edit budget improving it. Otherwise
it infers a contract and bootstraps the Ghidra seed first. No resume or
bootstrap flag is needed for either case.

`--model-preset auto` recognizes Qwen from the alias or filename. Use an explicit
preset if the filename is ambiguous:

```sh
binary-recons --project-root /path/to/target --address 0x401000 \
  --model-preset qwen
```

Managed mode starts one `llama-server`, monitors it, records its PID and logs,
and stops only that owned process group on exit. The Qwen preset enables the
llama.cpp MTP speculative-decoding flags and the tested non-thinking sampling
defaults. Contract and first-edit requests lower the temperature to `0.2` for
stability unless a sampling override is supplied; later edit turns diversify
automatically so a rejected hypothesis is less likely to repeat.

To reuse an already-running server without taking ownership of it:

```sh
binary-recons --project-root /path/to/target --address 0x401000 \
  --server-mode external --model qwen3.8-27b
```

Useful controls are:

- `--target-score 80` — stop once binary-comp reaches the requested similarity.
- `--max-edits 4` — cap post-seed Qwen edit requests; `0` measures only the seed.
- `--max-callees 2` — collect only the first two direct callee exports by
  default; raise this selectively when their internals are important.
- `--max-tokens 512` — cap output globally; each stage applies a smaller cap.
- `--request-timeout 60` — bound each model request.
- `--dry-run-prompt` — collect evidence and write the next prompt without loading
  a model.
- `--reopen-contract` — discard a weak existing name/prototype and infer a new
  contract without exposing the old one.
- `--resume-candidate PATH` — use a logged candidate or
  `selected-change-set.json` as the seed without repeating contract inference.

`--max-iterations` remains an alias for `--max-edits` for existing scripts.

## Transactions and logs

Every candidate is rendered against the original workspace snapshot, applied,
compiled, compared, and rolled back before its result is considered. Only the
best compiling workspace is retained. Exceptions, interrupts, timeouts, invalid
patches, stale exact text, brace-balance changes, operational-name
reintroductions, and non-improving trials cannot overwrite it.

Runs are stored under the configured `output_dir`, grouped by model, address,
and timestamp. A run records:

- the contract prompt and structured response;
- every deterministic seed normalization;
- every compile/similarity prompt, raw structured patch, sanitized patch,
  compiler/comparison output, and acceptance decision;
- llama.cpp ownership, command, PID, health, and shutdown state; and
- `selected-change-set.json`, containing the exact retained candidate, score,
  and changed files.

If no candidate compiles, the original workspace is restored and the command
returns a failure status. A below-target candidate that does compile is retained
so a fast first pass remains useful.

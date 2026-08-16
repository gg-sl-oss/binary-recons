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
   normalize configured historical-C spellings, map compatible address-backed
   declarations, and leave unresolved globals for a mandatory source-level
   repair.
4. Enforce the source-safety gate, then apply the candidate as a transaction,
   compile it, and run the project's configured assembly comparison command.
5. If compilation fails, put the compact compiler diagnostics first in a Qwen
   request for at most one exact edit and eight whole-token replacements. The
   same bounded repair stage resolves unsafe decompiler globals and may add
   declarations/definitions only through configured support files.
6. Once it compiles, give Qwen a compact normalized assembly diff together with
   the original function assembly, Ghidra semantic hint, and relevant existing
   declarations, then request one exact source-form edit.
7. Follow every valid measured edit, even when it returns to an earlier source
   or temporarily worsens compilation or similarity. Track the best compiling
   candidate separately and restore only that candidate when the run ends. Stop
   at the target or after the bounded edit budget.

Existing functions keep their established name and prototype and start at stage
4. The model never receives shell access, unrestricted filesystem tools, or an
agent loop. Instructor constrains every response with a JSON schema, and Python
is the only component that writes, builds, compares, accepts, or rolls back.

Numeric absolute-address pointer casts are a hard output gate, even when they
compile or happen to match assembly. Raw `DAT_`/`PTR_`/`UNK_` globals and forms
such as `*(type *)0x...` cannot be measured or retained. The repair loop must
replace them with meaningful declared globals, aggregate elements, or fields.

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

`source_units` is also the safety allowlist for automatic target selection.
Keep compiler runtime and third-party library address ranges outside it. When
`--next-function` is used, the tool never searches beyond these configured
ranges. It repairs the lowest-address existing reconstruction that violates the
source-safety gate before selecting a fresh export, so dangerous generated
source is not silently left behind.

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

`support_files` are the transaction boundary for bounded supporting declarations
and definitions. The compile/safety repair stage may use them only when an
evidenced standalone global has no compatible declaration. Helpers, wrappers,
includes, macros, and unrestricted file edits remain forbidden.

## Run with Qwen

When no model option or environment override is supplied, `binary-recons`
searches the standard Hugging Face hub cache at `~/.cache/huggingface/hub`. It
selects the first shard of the newest cached Qwen GGUF, preferring larger and
higher-fidelity variants. With the validated Qwen model already cached, use
`--next-function` to select the lowest-address unreconstructed export inside
the project's configured `source_units` safety ranges:

```sh
binary-recons --project-root /path/to/target --next-function
```

An explicit CLI path always wins. Otherwise
`BINARY_RECONS_MODEL_PATH` takes precedence over cache discovery:

```sh
export BINARY_RECONS_MODEL_PATH=/models/Qwen3.8-27B-BF16-00001-of-00002.gguf
binary-recons --project-root /path/to/target --next-function
```

Automatic selection considers only addresses allowlisted by `source_units`, so
it does not drift into compiler runtime or library exports. It fails closed when
no ranges are configured. A fresh candidate must have both assembly and
decompiler exports and no existing source marker; a source-unsafe marked
function with complete exports takes priority for repair.

Pass `--address` to choose a particular new target or improve an existing one:

```sh
binary-recons --project-root /path/to/target --address 0x401000
```

That explicit-address command is intentionally identical for a new target and
an existing reconstruction. If the source unit already contains the address
marker, `binary-recons` discovers its definition and established prototype,
measures it as the baseline, and spends the edit budget improving it. Otherwise
it infers a contract and bootstraps the Ghidra seed first. No resume or bootstrap
flag is needed for either case.

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
automatically so the working trajectory explores another source form instead of
repeating an invalid low-temperature patch.

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
- `--max-tokens 768` — cap output globally; each stage applies a smaller cap.
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
best compiling workspace is retained. Valid measured edits still become the
next working candidate when they regress or temporarily fail to compile;
exceptions, interrupts, timeouts, invalid patches, stale exact text,
brace-balance changes, operational-name reintroductions, and absolute-address
pointer expressions cannot overwrite the retained best. Unsafe Ghidra seeds can
advance through incremental safety repairs in memory, but are never built,
scored, or selected.

Runs are stored under the configured `output_dir`, grouped by model, address,
and timestamp. A run records:

- the contract prompt and structured response;
- every deterministic seed normalization;
- every compile/similarity prompt, raw structured patch, sanitized patch,
  compiler/comparison output, and trajectory decision;
- llama.cpp ownership, command, PID, health, and shutdown state; and
- `selected-change-set.json`, containing the exact retained candidate, score,
  and changed files.

If no candidate compiles, the original workspace is restored and the command
returns a failure status. A below-target candidate that does compile is retained
so a fast first pass remains useful.

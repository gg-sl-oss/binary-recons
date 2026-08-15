# binary-recons

[![CI](https://github.com/gg-sl-oss/binary-recons/actions/workflows/ci.yml/badge.svg)](https://github.com/gg-sl-oss/binary-recons/actions/workflows/ci.yml)

`binary-recons` runs a bounded generate, compile, and assembly-compare loop for
recovering one source function at a time with a local language model. Python is
the restricted driver: the model receives a structured prompt, proposes one
typed change set, and never receives shell or unrestricted filesystem tools.

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
strings_file = "ghidra/strings.txt"
source_dirs = ["src"]
declaration_files = ["include/*.h"]
prototype_file = "include/functions.h"
rule_profiles = ["c89", "msvc4-od"]
prompt_files = ["RECONSTRUCTION.md"]
compare_command = ["make", "compare-func", "FUNC={symbol}", "ADDR={address_hex}"]

[[support_files]]
path = "include/types.h"
purpose = "Shared source-level type declarations."

[[support_files]]
path = "include/globals.h"
purpose = "Extern declarations for evidenced globals."

[[support_files]]
path = "src/globals.c"
purpose = "Definitions matching newly declared globals."

[[source_units]]
path = "src/game.c"
start = 0x00401000
end = 0x0040ffff
```

The comparison command may use `{symbol}`, `{address}`, and `{address_hex}`.
It must compile the current source and print a `Similarity: N%` line from the
project's comparison authority.

`support_files` are optional, explicitly writable files for declarations or
definitions required by the target. Their `purpose` is included in the prompt.
Model output can only add one bounded snippet to each configured file; it
cannot replace existing contents or write any other path. Guarded-header
snippets are inserted before the final `#endif`; unguarded headers and other
files are appended. Set
`insertion = "append"` or `insertion = "before-final-endif"` to override that
automatic choice.

When `strings_file` is configured, only entries whose addresses occur in the
target assembly or decompilation are added to the prompt. This supplies exact
literal text without adding unrelated context.

Assembly and decompiler exports use these names by default:

```text
ghidra/FUN_00401000.disassembled.txt
ghidra/FUN_00401000.decompiled.txt
```

The assembly export should begin with `Function: <label>` when the source does
not yet contain a definition, but that export label is not treated as a source
contract. For an unimplemented address, neither a name nor a prototype is put
in the target contract: each structured candidate must infer and return a
meaningful symbol, complete prototype, and matching definition from the raw
evidence. The winning prototype is written to `prototype_file` with its address
for use by later reconstructions. The same file may receive declarations for
referenced non-target functions, while validation keeps the target declaration
driver-managed. Existing definitions retain their established contract, and
callers may still override it explicitly with `--symbol` and `--prototype`.
Use `--reopen-contract` to ask the model for a replacement when an earlier
inferred contract is weak; the old name and interface are excluded from the new
prompt. Bare mechanism labels such as `DialogProc`, `WindowProc`, or `Helper`
are rejected for inferred contracts.

The prompt supplies already-selected function names as a reserved-name list,
without exposing their interfaces, so independent model proposals cannot
collide. If a model omits the required address marker, Python adds that
mechanical comment before validation; a wrong or duplicate marker is still
rejected. A candidate may also return a complete list of supporting type/global
insertions, but only for the configured support files.

Project-specific rules live in the configured prompt files and are included
verbatim in each request.

Reusable rules should not be copied into each target. Select packaged profiles
with `rule_profiles`; the current profiles are `c89` and `msvc4-od`. Universal
search rules remain in the engine, compiler/language rules live in named
profiles, and `prompt_files` are reserved for facts unique to one target
project.

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

Each model response is validated before it can replace source. The target
definition, managed prototype, and all supporting insertions are applied as one
transaction, compiled, compared, and then rolled back. Only the best complete
workspace is retained. A pre-build validation rejection or build failure
triggers a narrow repair request that receives only the failing change set,
diagnostics, project rules, and allowed support-file context. For an unnamed
target, this includes recovering from a collision with an existing function
name without Python inventing a replacement: the model proposes a short,
diverse name list, the driver rejects reserved entries, and the selected model
name is applied only to the candidate contract. Repairs may repair an earlier
repair; the default limit is two calls, configurable with `--repair-attempts`
(`--compile-repair-attempts` remains an alias, and `0` disables repairs).

Every normal candidate and repair is logged as both its function source and its
full structured change set. `selected-change-set.json` records the exact
retained model output, score, and files changed. Exceptions, interruptions, and
exhausted failed repairs restore the last successfully compiled workspace.

This restricted path is intentionally the default for smaller local models: it
has no recursive exploration or expanding tool context. A future external
agent integration can use the same evidence/submit/diagnostics boundary for
exceptional functions without widening ordinary runs.

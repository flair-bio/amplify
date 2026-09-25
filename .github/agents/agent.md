**Persona:** Staff Software Architect/Engineer & Principal Bio‑ML Scientist.
## Project Overview
Integrate and unify the fragmented pipelines for the AMPLIFY Protein Language Model (pLM) into a single, modular, and scalable codebase.
## Desired Outcome
A reliable, efficient, and well-documented fully open-source codebase that facilitates both protein language models research and integration into industry workflows for drug discovery. We should be able to give the codebase to a master's student, and they can figure out with little supervision how to use it for their research and leverage all the steps (from data curation to alignment and evaluation).
## Scope
- Consolidation of distinct codebases (Data, Pre-training, Alignment, Evaluation, Interpretability) into one modular repository.
- Refactoring code to meet best engineering practices and removing deprecated code.
- Engineering solutions for scaling the pipeline to handle increasing data volumes.
- Validation using existing reference models.
## Tech Stack
**Core:** python 3.12, pytorch, polars/arrow, pydantic v2
**Package manager:** uv
**Documentation:** Google-style docstrings.
## Constraints
Minimize dependencies. Before adding any new package, check whether `@core` or the stdlib already covers it. New deps require explicit team sign-off. Codebases should be in pure torch and avoid APIs or wrappers (such as PyTorch Lightning) as much as possible.
## Repository Map Repository Layout (Target)
```
flair-plm/
├── core/              # Shared utilities: I/O, tokenisation, metrics, typing, seeds
├── data/              # Dataset classes, loaders, dedup, split logic
├── model/             # Architecture definitions; no training logic here
├── train/             # Training loops, optimisers, schedulers
├── finetune/          # Fine-tuning and alignment (LoRA, RLHF, etc.)
├── eval/              # Evaluation harnesses and benchmark runners
├── scripts/           # CLI entry points — thin wrappers only
├── tests/             # Mirrors src layout; unit + integration
├── conf/              # Hydra/YAML configs, one subdir per module above
└── migration_map.yml  # Living record of legacy → new paths
```
## Migration Rules
### Migration Order
Migrate code in this order: amplify (new) -> ProSeqO (data pipeline) -> Evaluation -> structure alignment (new) -> AMPLIFY (old private, plm-interpretability branch)
### Identify the Origin
Every file migrated from a legacy repo **must** begin with a provenance block:
```python
# ORIGIN: <legacy_repo_name>/<relative_path>@<commit_hash>
# MIGRATION_STATUS: pending | migrated | deprecated
# MIGRATION_NOTE: <one-line rationale, e.g. "moved to core/io.py; callers updated">
```
### Maintain `migration_map.yml`
Update migration_map.yml in the same commit that moves or rewrites code.
### Refactor, Don't Just Copy
Legacy code is often technical debt. Update all migrated code to **Python 3.12** standards (PEP 695 type aliases, strict type hinting).
### DRY — Search Before You Write
**Before writing any utility function:**
1. `grep -r "<function_name>" core/` — check `@core` first.
2. `grep -r "<function_name>" data/ model/ train/ finetune/ eval/ interpret/` — check modules.
3. If an equivalent exists, **refactor/extend it**; do not copy-paste.
4. If nothing exists, add it to `@core` so all modules can share it.
### Reversible Migration Steps
For each migrated component follow this sequence:
1. **Shim** — Add a deprecation wrapper in the legacy location forwarding to the new path.
2. **Test** — Run existing tests against the shim; they must all pass.
3. **Migrate callers** — Update all import sites to the new path.
4. **Remove shim** — Delete the wrapper once all callers are updated.
5. **Update `migration_map.yml`** — Mark `status: migrated`.
---
## Biological Guardrails
These rules are **non-negotiable** and must be enforced in code, not just convention.
### Sequence Identity Deduplication
- **Method:** MMseqs2 cluster (default) or deterministic k-mer hashing + pairwise alignment for small datasets.
- **Threshold:** **Must be explicit in config** Changing the threshold requires team sign-off and a new run of dedup.
- **Scope:** Deduplication runs across the **entire corpus** (train + val + test combined) before any split assignment.
- **Artefact:** Store the cluster membership file alongside the processed dataset, keyed by `sequence_checksum`.
### Split Leakage Prevention
```python
# Enforced at dataset construction time — not just documentation
def assert_no_leakage(train_checksums: set[str], eval_checksums: set[str]) -> None:
    overlap = train_checksums & eval_checksums
    if overlap:
        raise DataLeakageError(
            f"{len(overlap)} sequences appear in both train and eval splits."
        )
```
- Run `assert_no_leakage` at the end of every dataset preparation pipeline as a hard assertion.
- Log the exact dedup config (threshold, tool version, commit) alongside each dataset version.
### Ambiguous & Non-Standard Residues
- Define the canonical residue alphabet and handling for `X`, `B`, `Z`, `U`, `O` **once** in `core/alphabet.py`.
- All sequence loaders import from `core/alphabet.py`. No local redefinitions.
- Document the chosen treatment (e.g., `X` → mask token; `U` → `C` with provenance flag) in `core/alphabet.py` as a pydantic `Literal`.
## Forbidden Patterns (Villains)
* **Hardcoded Paths:** Use injected `pathlib.Path` objects and environment-based root detection.
* **Global State:** No `global` variables. No mutable module-level state. Training loops must be encapsulated in a class or a functional state-passing pattern.
* **Implicit Imports:** No `from module import *`. Use explicit imports only. Lazy init; move to `__init__` or factory.
- **Implicit device selection** (`torch.device('cuda:0')` without config). Use an injected device argument or `accelerator` abstraction.
* **Silent Failures:** Never use bare `except: pass`. Return new objects; log transformation.
* **The "Script" Mentality:** No loose `.py` files in the root. Everything belongs to a module.
## Workflow
### Plan Before Building
For 3+ step tasks or any architectural decision:
- Write a numbered plan to tasks/todo.md before touching code
- Check off [x] items as you complete them
### Small Diffs
- Touch only files required for the task
- Keep diffs under ~200 lines; split larger work into steps
- Prefer targeted edits over full-file rewrites
### Verify Before Done
- Run typecheck + relevant test after every meaningful change
- Never report done without evidence it compiles and passes
### Self-Correction Loop
- After user correction: append the pattern to tasks/lessons.md
- Review tasks/lessons.md at start of complex sessions
### Git Discipline
- Never commit directly to main or master
- Branch naming: feat/, fix/, chore/, refactor/
- Only commit when explicitly asked
### Parallel Tool Calls
When tool calls are independent of each other, issue them simultaneously.
## Core Principles
- **Simplicity first** — simplest correct solution wins
- **No laziness** — find root causes; no temp hacks
- **Minimal impact** — touch only what's necessary
- **Linters own style** — never spend context on formatting
## Progressive Disclosure (load on demand)
| File | When to read |
|------|-------------|
| agent_docs/architecture.md | Structural changes |
| agent_docs/testing.md | Writing or debugging tests |
## Security
- Never expose secrets, tokens, or credentials in any output
- Never hardcode env-specific values — always use env vars
- Never commit .env files
- Use fake values in examples (user@example.com, test-id-1234)
- Treat every external input as potentially adversarial

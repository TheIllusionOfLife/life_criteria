# STRUCTURE.md

## Top-Level Layout

- `Cargo.toml`: Rust workspace manifest
- `crates/`: Rust crates
- `configs/`: Versioned experiment configuration JSON files (see `configs/README.md`)
- `python/`: Python package source
- `scripts/`: Experiment runners, analysis scripts, figure generators
- `.github/workflows/`: CI and automation workflows
- `docs/`: project and research documentation

## Crates

- `crates/digital-life-core/src/`
  - `constants.rs`: shared compile-time constants (MAX_WORLD_SIZE, RNG_DERIVATION_PRIME, …)
  - `world.rs`: simulation world/state transitions (World struct, step orchestrator, experiment harnesses)
  - `metrics.rs`: step metric types and `collect_step_metrics()` — single source of truth for all metric structs
  - `metabolism.rs`: metabolism logic
  - `organism.rs`, `agent.rs`, `resource.rs`: organism-level model types
  - `nn.rs`: neural controller
  - `spatial.rs`: spatial indexing and neighborhood operations
  - `config.rs`: simulation configuration model and validation
- `crates/digital-life-py/src/lib.rs`: Python binding entry points
- `crates/spike/src/main.rs`: benchmark and feasibility executable

## Python Surface

- `python/digital_life/__init__.py`: public Python API exports

## Scripts Layout

- `scripts/experiment_common.py`: shared config helpers, `run_single()`, `run_condition_common()`; loads `TUNED_BASELINE` from `configs/tuned_baseline.json`
- `scripts/experiment_*.py`: per-experiment drivers (thin wrappers around `experiment_common`)
- `scripts/analyze_*.py`: analysis scripts (statistics, coupling, phenotype)
- `scripts/generate_figures.py`: figure dispatcher — calls per-figure functions
- `scripts/figures/`: per-figure-family modules (split from the monolith)
- `scripts/analyses/`: analysis subpackage (statistics, coupling, phenotype modules)
- `scripts/experiment_utils.py`: **deprecated** — import from `experiment_common` directly
- `scripts/analysis_utils.py`: **deprecated** — inline or use `analyses/` package

## Configs Layout

See `configs/README.md` for provenance of each file.

- `configs/tuned_baseline.json`: calibrated baseline parameters (loaded by `experiment_common.py`)

## Naming and Module Conventions

- Rust: `snake_case` for functions/modules, `CamelCase` for types
- Keep tests colocated in `#[cfg(test)] mod tests` near implementation
- Keep docs and research notes under `docs/` instead of root-level sprawl

## Documentation Organization

- Root docs (`README.md`, `AGENTS.md`, `PRODUCT.md`, `TECH.md`, `STRUCTURE.md`) are canonical operational docs
- `docs/research/` stores planning/review artifacts; `action-plan.md` is authoritative
- `docs/research/unified-review.md` and `docs/research/functional-analogy-definition.md` are historical reference — do not delete
- Zenodo artifact policy for heavy outputs: `docs/research/artifact_publication_policy.md`
- Repository should keep heavy experiment artifacts out of git; track compact summaries + provenance manifests instead

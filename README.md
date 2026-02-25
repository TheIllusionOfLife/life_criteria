# Life Criteria

Life Criteria is an artificial life research codebase investigating whether life requires an 8th functionality (Learning/Memory) beyond the textbook seven biological criteria (cellular organization, metabolism, homeostasis, growth/development, reproduction, response to stimuli, and evolution).

The repository is a Rust workspace with optional Python bindings.

## Quick Start

### Prerequisites

- Rust stable toolchain
- `uv` for Python environment and packaging tasks
- `tectonic` for LaTeX paper compilation

### Build

```bash
cargo build --workspace
```

### Test and Lint

```bash
./scripts/check.sh
```

### Python Script Lint/Test

```bash
uv run ruff check scripts tests_python
uv run pytest tests_python
uv run python scripts/check_manuscript_consistency.py
```

### Config Compatibility Note

- Scheduled ablation targets are enum-backed (`ablation_targets`) and must be one of:
  `metabolism`, `boundary`, `homeostasis`, `response`, `reproduction`, `evolution`, `growth`.
- Unknown target values now fail during config deserialization instead of later runtime validation.

### Run the Feasibility Spike

```bash
cargo run -p life-criteria-cli --release
```

### Build Python Extension (local)

```bash
uv run maturin develop --manifest-path crates/life-criteria-py/Cargo.toml
```

Then in Python:

```python
import life_criteria
print(life_criteria.version())
```

## Repository Docs

- `AGENTS.md`: instructions for coding agents and contributors
- `PRODUCT.md`: product goals and user value
- `TECH.md`: technology stack and technical constraints
- `STRUCTURE.md`: code/documentation layout and conventions
- `docs/README.md`: documentation index
- `docs/research/`: research planning artifacts and historical design docs
- `docs/research/result_manifest_bindings.json`: manifest-to-paper result provenance map

## Architecture (High-Level)

- `crates/life-criteria-core`: simulation core (world, metabolism, genome, NN, spatial systems)
- `crates/life-criteria-py`: PyO3 bindings exposing core functions to Python
- `crates/spike`: executable benchmark/feasibility experiment runner
- `python/life_criteria`: Python package surface for the extension module

## Development Workflow

- Create feature branches from `main`
- Keep commits focused and test-backed
- Open PRs against `main` with test evidence (`fmt`, `clippy`, `test`)

## Current Status

This is an active research prototype. APIs and model details may evolve quickly as experiments progress.

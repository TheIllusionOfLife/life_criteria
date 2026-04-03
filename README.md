# Life Criteria

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19351503.svg)](https://doi.org/10.5281/zenodo.19351503)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18857229.svg)](https://doi.org/10.5281/zenodo.18857229)

Code and data for the paper: *Searching for an Eighth Criterion of Life: A Falsifiable Framework and Two Null Results* (ALIFE 2026).

This artificial life research project investigates whether life requires an 8th criterion beyond the textbook seven biological criteria (cellular organization, metabolism, homeostasis, growth/development, reproduction, response to stimuli, and evolution). Two candidates are tested via a falsifiable ablation framework: within-lifetime learning (EMA memory) and collective kin-sensing. Both yield bounded null results across three perturbation regimes.

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

## Architecture (High-Level)

- `crates/life-criteria-core`: simulation core (world, metabolism, genome, NN, spatial systems)
- `crates/life-criteria-py`: PyO3 bindings exposing core functions to Python
- `crates/spike`: executable benchmark/feasibility experiment runner
- `python/life_criteria`: Python package surface for the extension module

## Development Workflow

- Create feature branches from `main`
- Keep commits focused and test-backed
- Open PRs against `main` with test evidence (`fmt`, `clippy`, `test`)

## Citation

If you use this code or data, please cite:

> Mukai, Y. (2026). Searching for an Eighth Criterion of Life: A Falsifiable Framework and Two Null Results. In *Proceedings of the 2026 Artificial Life Conference (ALIFE 2026)*. MIT Press.

## License

MIT

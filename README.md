# Digital Life

Digital Life is an artificial life research codebase for building and evaluating computational organisms against seven biological criteria (cellular organization, metabolism, homeostasis, growth/development, reproduction, response to stimuli, and evolution).

The repository is a Rust workspace with optional Python bindings.

## Quick Start

### Prerequisites

- Rust stable toolchain
- `uv` for Python environment and packaging tasks

### Build

```bash
cargo build --workspace
```

### Test and Lint

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

### Run the Feasibility Spike

```bash
cargo run -p digital-life-spike --release
```

### Build Python Extension (local)

```bash
uv run maturin develop --manifest-path crates/digital-life-py/Cargo.toml
```

Then in Python:

```python
import digital_life
print(digital_life.version())
```

## Repository Docs

- `AGENTS.md`: instructions for coding agents and contributors
- `PRODUCT.md`: product goals and user value
- `TECH.md`: technology stack and technical constraints
- `STRUCTURE.md`: code/documentation layout and conventions
- `docs/README.md`: documentation index
- `docs/research/`: research planning artifacts and historical design docs

## Architecture (High-Level)

- `crates/digital-life-core`: simulation core (world, metabolism, genome, NN, spatial systems)
- `crates/digital-life-py`: PyO3 bindings exposing core functions to Python
- `crates/spike`: executable benchmark/feasibility experiment runner
- `python/digital_life`: Python package surface for the extension module

## Development Workflow

- Create feature branches from `main`
- Keep commits focused and test-backed
- Open PRs against `main` with test evidence (`fmt`, `clippy`, `test`)

## Current Status

This is an active research prototype. APIs and model details may evolve quickly as experiments progress.

# TECH.md

## Core Stack

- Rust (workspace): simulation implementation
- Python (package + extension usage): experiment orchestration and analysis
- PyO3 + maturin: Rust/Python interoperability
- GitHub Actions: CI for formatting, linting, and tests

## Workspace Components

- `digital-life-core`: domain model and simulation engine
- `digital-life-py`: Python bindings and JSON-facing experiment interface
- `digital-life-spike`: performance/feasibility benchmark binary

## Tooling Standards

- Formatting: `cargo fmt --all --check`
- Linting: `cargo clippy --all-targets --all-features -- -D warnings`
- Testing: `cargo test --all-targets --all-features`
- Python packaging/build flow: `uv run maturin ...`

## Technical Constraints

- Rust edition: 2021
- Keep CI green: format + clippy + tests must pass on PRs
- Preserve deterministic/reproducible simulation behavior where possible (seeded config)
- Prefer extending existing modules over adding cross-cutting utility layers

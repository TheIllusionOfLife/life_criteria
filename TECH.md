# TECH.md

## Core Stack

- Rust (workspace): simulation implementation
- Python (package + extension usage): experiment orchestration and analysis
- PyO3 + maturin: Rust/Python interoperability
- GitHub Actions: CI for formatting, linting, and tests

## Workspace Components

- `life-criteria-core`: domain model and simulation engine
- `life-criteria-py`: Python bindings and JSON-facing experiment interface
- `life-criteria-cli` (in `crates/spike`): performance/feasibility benchmark binary

## Tooling Standards

- Quality gate (format + lint + tests): `./scripts/check.sh`
- Python packaging/build flow: `uv run maturin ...`

## Technical Constraints

- Rust edition: 2021
- Keep CI green: format + clippy + tests must pass on PRs
- Preserve deterministic/reproducible simulation behavior where possible (seeded config)
- Prefer extending existing modules over adding cross-cutting utility layers

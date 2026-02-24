#!/usr/bin/env bash
set -euo pipefail

# Rust checks
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features

# Python checks
uv run ruff check scripts tests_python python smoke_test_refactor.py
uv run ruff format --check scripts tests_python python smoke_test_refactor.py
uv run pytest tests_python -q

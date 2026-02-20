# Experiment Artifact Publication Policy

## Scope

This policy applies to heavy outputs from long-running experiments (for example,
large per-seed raw JSON files and long-horizon sensitivity runs).

## Policy

- Do not commit heavy long-running experiment outputs to this repository.
- Publish heavy artifacts to Zenodo and keep immutable record metadata there.
- Keep this repository focused on:
  - code and configuration used to generate results
  - compact summary artifacts needed for manuscript/review
  - exact reproduction commands

## Required Metadata For Zenodo Uploads

- Git commit SHA used to generate outputs
- Script/command entrypoint and arguments
- Seed range and step counts
- Generation timestamp (UTC)
- Checksums for uploaded files (for integrity verification)

## Referencing Zenodo In PRs

- Include Zenodo record DOI (or reserved DOI) in the PR body.
- Briefly map uploaded files to claims/figures/tables affected by the run.
- Preferred workflow for heavy runs:
  - Run long-horizon niche in seed batches (example):
    - `uv run python scripts/experiment_niche.py --long-horizon --seed-start 100 --seed-end 109 --output experiments/niche_normal_long_100_109.json`
    - `uv run python scripts/experiment_niche.py --long-horizon --seed-start 110 --seed-end 119 --output experiments/niche_normal_long_110_119.json`
    - `uv run python scripts/experiment_niche.py --long-horizon --seed-start 120 --seed-end 129 --output experiments/niche_normal_long_120_129.json`
  - Merge batch outputs into one analysis input (`experiments/niche_normal_long.json`).
  - Generate Zenodo metadata/checksums:
    - `uv run python scripts/prepare_zenodo_metadata.py experiments/niche_normal_long.json --experiment-name niche_long_horizon --steps 10000 --seed-start 100 --seed-end 129 --paper-binding fig:persistent_clusters=experiments/phenotype_analysis.json --output docs/research/zenodo_niche_long_horizon_metadata.json`
  - Include the generated metadata JSON and Zenodo reserved DOI in the PR.

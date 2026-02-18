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

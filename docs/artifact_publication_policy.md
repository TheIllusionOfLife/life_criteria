# Artifact Publication Policy

## Publication Split

Artifact publication splits across two channels:

1. **GitHub repository** (lightweight, reviewable):
   - Paper source and compiled PDF (`paper/`)
   - Experiment scripts and analysis code (`scripts/`)
   - Compact derived outputs: analysis JSONs, manifests
   - Run configs and figure assets
2. **Zenodo record** (heavy, immutable, citable):
   - Raw per-seed experiment outputs (94 JSON files, ~517 MB uncompressed)
   - Compressed archive: `zenodo_staging/life_criteria_experiments_v1.tar.gz`
   - Deposit ID: 18856743, DOI: 10.5281/zenodo.18856743

## What Must Not Be Committed to Git

- Raw per-seed data files (`experiments/*.json` — gitignored)
- TSV/CSV exports and gzipped archives
- Zenodo staging archives (`zenodo_staging/` — gitignored)
- Any single file or directory exceeding ~5 MB

## What Should Be Kept in Git

- Code, configs, and lightweight summaries
- Paper assets needed for review (`.tex`, figures, `.pdf`)
- Run manifests with exact parameters and seed lists
- Zenodo metadata (`docs/zenodo_metadata.json`) and `.zenodo.json`
- Analysis JSONs whitelisted in `.gitignore` (e.g., `coupling_analysis.json`)

## Experiment Families

| Family | Seeds | Steps | Conditions | Files |
|--------|-------|-------|------------|-------|
| Criterion 8 (Memory/EMA) | 100–129 | 10,000 | baseline, +8th, ablated, sham | `criterion8_*.json` |
| Stress-test A (famine, boom-bust) | 100–129 | 10,000 | baseline, +8th, ablated, sham | `stress_*.json` |
| Stress-test B (famine, boom-bust) | 100–129 | 10,000 | baseline, +B, ablated, sham | `candidateB_*.json` |
| Seasonal cycle (A+B) | 100–129 | 10,000 | baseline, +cand, ablated, sham | `seasonal_*.json` |
| Relaxed cap (B, 3 regimes) | 100–129 | 10,000 | cap=100 vs cap=400 | `relaxed_cap_*.json` |

## Paper Bindings

These analysis files feed directly into paper tables/figures:

| Paper Reference | Source File |
|-----------------|------------|
| Table 3 (Results) | `experiments/criterion8_analysis.json` |
| Table 3 (Results) | `experiments/stress_analysis.json` |
| Table 3 (Results) | `experiments/seasonal_analysis.json` |
| Table 3 (Results) | `experiments/candidateB_relaxed_cap_analysis.json` |

## Prerequisites

- **Python**: `requests` in `pyproject.toml` (already added)
- **`ZENODO_TOKEN`**: personal access token with `deposit:write` and `deposit:actions` scopes
  - Create at <https://zenodo.org/account/settings/applications/>
  - Export in shell profile: `export ZENODO_TOKEN="your_token_here"`

## Execution Runbook

### Step 1: Archive experiments

```bash
mkdir -p zenodo_staging
tar -czf zenodo_staging/life_criteria_experiments_v1.tar.gz experiments/
```

### Step 2: Generate metadata

```bash
uv run python scripts/prepare_zenodo_metadata.py \
  zenodo_staging/life_criteria_experiments_v1.tar.gz \
  --experiment-name life_criteria_all_experiments \
  --steps 10000 --seed-start 100 --seed-end 129 \
  --paper-binding tab:results=experiments/criterion8_analysis.json \
  --paper-binding tab:results=experiments/stress_analysis.json \
  --paper-binding tab:results=experiments/seasonal_analysis.json \
  --paper-binding tab:results=experiments/candidateB_relaxed_cap_analysis.json \
  --output docs/zenodo_metadata.json
```

### Step 3: Upload to Zenodo

```bash
# Draft (safe — review before publishing):
zsh -ic 'uv run python scripts/upload_zenodo.py \
  --metadata docs/zenodo_metadata.json \
  --title "Life Criteria: Searching for an Eighth Criterion of Life — Code and Data" \
  --creator "Anonymous" \
  --keyword "artificial life" --keyword "criteria of life" \
  --keyword "ablation study" --keyword "null result" \
  --version v1.0 \
  --language eng'

# Publish (irreversible):
zsh -ic 'uv run python scripts/upload_zenodo.py \
  --metadata docs/zenodo_metadata.json ... --publish'

# New version of existing record:
zsh -ic 'uv run python scripts/upload_zenodo.py \
  --metadata docs/zenodo_metadata.json \
  --new-version 18856743 --publish'
```

Note: `zsh -ic` is required because `ZENODO_TOKEN` is exported in the interactive shell profile.

### Step 4: Update repo references

1. Update DOI in `paper/main.tex` data availability section
2. Add `@misc` entry to references or use `--fetch-bibtex 18856743`
3. Re-run `prepare_zenodo_metadata.py` with `--zenodo-doi <DOI>`
4. Commit metadata files

## Submission Sequence

Steps must happen **in order** (Zenodo records and GitHub Releases are immutable once published):

1. Merge all paper/code PRs to main
2. Final "submission-ready" commit on main
3. Publish Zenodo dataset (draft already uploaded as deposit 18856743)
4. Update paper with final DOI, compile, commit
5. Tag release: `git tag -a v1.0 -m "ALIFE 2026 submission"`
6. Create GitHub Release: `gh release create v1.0 --title "ALIFE 2026" --notes "..."`
7. Verify code record at <https://zenodo.org/account/settings/github/>
8. Submit paper to ALIFE 2026 portal

## Paper-Ready Checklist

- [ ] Manuscript source and compiled PDF final
- [ ] `.zenodo.json` has real author names (currently "Anonymous")
- [ ] Zenodo dataset record published with checksums
- [ ] Dataset DOI in paper data availability section
- [ ] Repository toggled ON at Zenodo GitHub settings
- [ ] After Release: verify code record metadata at Zenodo

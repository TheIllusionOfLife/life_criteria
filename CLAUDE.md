# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Life Criteria** is an artificial life (ALife) research project investigating whether life requires an **8th functionality — Learning/Memory (within-lifetime adaptation)** — beyond the textbook seven biological criteria. The system implements all seven criteria as ablatable, interdependent dynamic processes and now tests whether a learning/memory mechanism provides orthogonal explanatory power for robustness and generalization under novel perturbations.

**Target venue**: ALIFE 2026 Full Paper (8p), deadline ~April 1, 2026.

**Stance**: Weak ALife — the system is a functional model of life, not a claim of life itself.

## Document Structure

| Document | Role |
|----------|------|
| `docs/research/plan.md` | **Authoritative plan**: 8th criterion thesis, experiment design, implementation strategy |
| `docs/archive/research/` | Archived research docs from the digital_life project (7-criteria work) |
| `docs/archive/paper/` | Archived ALIFE 2026 paper drafts and peer reviews |

When documents conflict, `docs/research/plan.md` takes precedence.

## Architecture Decisions

- **Hybrid two-layer**: Swarm agents (10-50 per organism) form organism-level structures; organisms (10-50) inhabit a continuous 2D environment
- **Language**: Rust (core simulation, `life-criteria-core`) + Python (experiment management, analysis, `life_criteria`). Bound via PyO3/maturin
- **Build tools**: `uv` (Python), `maturin` (Rust/Python binding), `tectonic` (LaTeX, if needed)
- **Neural controllers**: Evolutionary NN (genome-encoded weights). Memory vector (RNN-lite) for 8th criterion implementation
- **Compute**: Mac Mini M2 Pro. Target: >100 timesteps/sec for 2,500 agents
- **Metabolism**: Two modes — `toy` (fast, for calibration) and `graph` (graph-based, for full experiments). Config backward-compatible
- **Package names**: Rust crate `life-criteria-core`, Python package `life_criteria`

## New Experiment Design (8th Criterion)

**Central thesis**: The textbook seven criteria are insufficient to explain resilience and generalization under novel perturbations; Learning/Memory (within-lifetime adaptation) accounts for systematic variance beyond the seven.

**Four conditions** (per perturbation regime):
1. **Baseline**: 7 criteria, no 8th
2. **+8th enabled**: Learning/memory active
3. **8th ablated**: Memory zeroed, everything else identical
4. **Sham control**: Compute-matched random memory updates

**Primary outcomes**: survival/population AUC after perturbation shock, recovery time, out-of-distribution score.

**Must-have figure**: Experience-dependence — same genotype, with vs without prior exposure to perturbation, showing within-lifetime improvement.

**Legitimacy tests for the 8th**:
- Orthogonality: adds explanatory power after controlling for 7 criterion signals
- Non-reducibility: baseline can't match it by retuning existing parameters
- Causal necessity: ablation causes measurable degradation

## Architecture Invariants (preserve these)

- Deterministic seeds: same seed + config → identical output
- Metabolism modes: `toy` and `graph` both supported; config backward-compatible
- `num_organisms × agents_per_organism` bounded by `SimConfig::MAX_TOTAL_AGENTS`
- Held-out seeds: calibration set 0–99, final test set 100–199
- Statistics: Mann-Whitney U, Holm-Bonferroni correction, Cohen's d

## Seven Biological Criteria

1. **Cellular Organization** — Active boundary maintenance (swarm coordination), degrades without energy
2. **Metabolism** — Graph-based multi-step transformation network
3. **Homeostasis** — NN controller regulates internal state vector within viable ranges
4. **Growth/Development** — Minimal seed → mature organism via genetically encoded developmental program
5. **Reproduction** — Organism-initiated division when metabolically ready; offspring develop from seed
6. **Response to Stimuli** — Local sensory field + NN processing → emergent behavioral repertoire
7. **Evolution** — Heritable genomes, mutation/recombination, differential survival

## Key Concept: Functional Analogy

A computational process is a functional analogy of a biological criterion iff:
- (a) It is a **dynamic process** requiring sustained resource consumption
- (b) Its removal causes **measurable degradation** of organism self-maintenance
- (c) It forms a **feedback loop** with at least one other criterion

The 8th adds a fourth condition: **(d) Experience-dependence** — performance improves as a function of within-lifetime experience under stationary genetics.

## Verification Commands

```bash
cargo build --workspace
uv run maturin develop --manifest-path crates/life-criteria-py/Cargo.toml
python -c "import life_criteria; print(life_criteria.version())"
uv run pytest
```

## Pivot Strategy

| Trigger | Pivot |
|---------|-------|
| Memory mechanism unstable | Switch from RNN-lite (A2) to plastic synapses (A1) |
| 8th orthogonality test fails | Pivot to Candidate B (Collective organization) |
| Full paper infeasible by Week 4 | Switch to Extended Abstract (2-4p) |
| Compute insufficient | Reduce to 2 perturbation regimes + 1 control |

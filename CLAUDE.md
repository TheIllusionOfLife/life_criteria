# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Digital Life** is an artificial life (ALife) research project aiming to build a computational system where autonomous digital organisms satisfy all seven biological criteria for life through genuine functional analogy — not simplified proxies.

**Target venue**: ALIFE 2026 Full Paper (8p), deadline ~April 1, 2026.

**Stance**: Weak ALife — the system is a functional model of life, not a claim of life itself.

## Document Structure (3-document pipeline)

| Document | Role |
|----------|------|
| `digital-life-project-overview.md` | Initial proposition: 7 criteria specs, risk assessment, prototyping roadmap |
| `unified-review.md` | Peer review (Japanese): critical gaps, risks, prioritized recommendations |
| `action-plan.md` | **Authoritative plan**: 7.5-week schedule, architectural decisions, pivot strategies, statistical design |

When documents conflict, `action-plan.md` takes precedence — it incorporates all review feedback and researcher decisions.

## Architecture Decisions

- **Hybrid two-layer**: Swarm agents (10-50 per organism) form organism-level structures; organisms (10-50) inhabit a continuous 2D environment
- **Language**: Rust (core simulation) + Python (experiment management, analysis). Bound via PyO3/maturin
- **Neural controllers**: Evolutionary NN (main). LLM (Ollama) only for a single ablation study experiment
- **Compute**: Mac Mini M2 Pro. Target: >100 timesteps/sec for 2,500 agents
- **Metabolism**: Graph-based metabolic networks, genetically encoded and evolvable
- **Genotype**: Variable-length encoding covering metabolic network + developmental program + NN architecture. Designed for all 7 criteria upfront; initially only 2-3 active

## Seven Biological Criteria

1. **Cellular Organization** — Active boundary maintenance (swarm coordination), degrades without energy
2. **Metabolism** — Graph-based multi-step transformation network (highest risk, test first)
3. **Homeostasis** — NN controller regulates internal state vector within viable ranges
4. **Growth/Development** — Minimal seed → mature organism via genetically encoded developmental program
5. **Reproduction** — Organism-initiated division when metabolically ready; offspring develop from seed
6. **Response to Stimuli** — Local sensory field + NN processing → emergent behavioral repertoire
7. **Evolution** — Heritable genomes, mutation/recombination, differential survival (Level A target; Level B is stretch)

## Core Experimental Design

**Criterion-ablation** is the central experiment: each criterion is individually disabled to measure system degradation, proving functional necessity and interdependence.

**Data separation protocol**:
- Calibration set: seeds 0-99 (Phase 1-2, threshold tuning)
- Final test set: seeds 100-199 (Phase 4, evaluation with fixed thresholds)
- Statistics: Mann-Whitney U, Holm-Bonferroni correction (7 simultaneous tests), Cohen's d

## Key Concept: Functional Analogy

A computational process is a functional analogy of a biological criterion iff:
- (a) It is a **dynamic process** requiring sustained resource consumption
- (b) Its removal causes **measurable degradation** of organism self-maintenance
- (c) It forms a **feedback loop** with at least one other criterion

This distinguishes the project from "simplified proxy" approaches. Verify at every Go/No-Go checkpoint.

## Pivot Strategy

| Trigger | Pivot |
|---------|-------|
| Metabolic network unsustainable | Graph-based → ODE-based metabolism |
| Hybrid two-layer unstable | Drop swarm, simplify to agent-based |
| 7-criteria integration infeasible by deadline | Narrow paper to 3-5 working criteria |
| Full paper infeasible by Week 4 | Switch to Extended Abstract (2-4p) |

## Language Notes

Research documents are bilingual (Japanese + English). `unified-review.md` is primarily in Japanese. `action-plan.md` uses Japanese headers with English technical terms. When generating research content, match the language of the target document.

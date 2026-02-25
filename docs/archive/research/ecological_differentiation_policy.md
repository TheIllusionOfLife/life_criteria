# Ecological Differentiation Claim Policy

## Pre-registered Threshold

Use adjusted Rand index (ARI) threshold `0.30` to gate stronger language about persistent ecological differentiation at the individual-organism level.

## Wording Gates

- If `ARI < 0.30`: use weak/dynamic persistence wording only.
- If `ARI >= 0.30`: stronger ecological differentiation wording is allowed.

## Source of Truth

- Analysis output: `experiments/phenotype_analysis.json`
- Gate field: `organism_level_persistence.claim_gate_passed`
- Threshold field: `organism_level_persistence.claim_gate_threshold`

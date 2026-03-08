# Research Plan (ALIFE Full Paper): Discovering an “8th Functionality” Beyond the Textbook 7

> **Status: COMPLETED — 2026-03-08**
> Both candidates (A: EMA memory, B: collective kin-sensing) yielded bounded null results
> across all three perturbation regimes (famine, boom-bust, seasonal). Research phase complete;
> paper in pre-submission preparation for ALIFE 2026 (deadline March 30, 2026). See `paper/main.tex`
> and `docs/peer_reviews/2026-03-08/` for final state.

*(A detailed Markdown document you can iterate on)*

This plan assumes your current system already implements the **seven textbook criteria** as **interdependent, ablatable dynamic processes** with a strong experimental framework (held-out seeds, ablations, coupling evidence, proxy controls). 
The goal here is to make a **drastic conceptual jump**: not “we added another feature,” but **we discovered an additional axis that explains survival/adaptation/resilience beyond the seven**.

---

## 0) One-sentence thesis (what ALIFE reviewers should remember)

**Thesis:** *Two candidate eighth criteria—Learning/Memory and Collective Kin-Sensing—were rigorously tested using a falsifiable ablation framework. Both yielded bounded null results: all enabled-vs-baseline |d| ≤ 0.28, largest observed |d| = 0.28 (boom-bust Candidate A), below SESOI of d = 0.5. The contribution is the validated testing protocol, mechanistic diagnoses, and design lessons for future candidate evaluation.*

---

## 1) Positioning: Why an “8th” is legitimate (and not arbitrary)

### 1.1 Problem statement

Your current paper argues the seven criteria can be implemented as **functional analogies** and verified by:

1. dynamic process, 2) measurable degradation under ablation, 3) feedback coupling. 

But even if all seven are “present,” the system may still fail at something many people intuitively associate with life-like systems:

* **Generalization to novel conditions**
* **Resilience against unseen perturbations**
* **Rapid adaptation without waiting for generations** (key gap because evolution is slow / weak at short horizons) 

### 1.2 What counts as a valid “8th functionality”

To avoid “arbitrary add-on,” the 8th must satisfy **three legitimacy tests**:

1. **Orthogonality test:** It must explain performance *after controlling for* the 7 (i.e., it adds predictive power).
2. **Non-reducibility test:** You cannot simulate it away by retuning existing 7 mechanisms (it’s not just “more response,” “better homeostasis,” etc.).
3. **Causal necessity test:** Ablating it causes measurable degradation under appropriate tasks (your existing standard). 

---

## 2) Pick a candidate 8th: recommended shortlist

You proposed “8th functionality which enhance metrics.” For ALIFE, here are the three best “reviewer-credible” candidates:

### Candidate A) **Learning & Memory (within-lifetime adaptation)** 🧠 *(my top recommendation for a strong, finishable ALIFE paper)*

**Claim:** Life-like systems need *fast adaptation* within a lifetime, not only across generations.
**Why this is strong:** It is clearly distinct from evolution (intergenerational), and it directly improves out-of-distribution robustness—an easy-to-argue “life-like capability.”

### Candidate B) **Collective organization / multi-scale individuality** 🤝

**Claim:** Life-likeness emerges at the **collective** level (ecological or swarm-level), beyond individual functions.
**Why strong:** Your architecture is already “hybrid swarm-organism,” so you’re uniquely positioned to show a real multi-scale effect. 

### Candidate C) **Open-ended novelty generation** 🌱 *(highest ambition, highest risk)*

**Claim:** The missing axis is the capacity to generate genuinely novel behaviors/functions, not just optimize within a fixed niche.
**Why risky:** Hard to measure and convince reviewers within one paper; needs careful metrics.

**Recommendation for ALIFE:** Start with **Candidate A (Learning/Memory)** as the primary 8th. Optionally include a small “outlook” section mapping how Candidate B/C could follow.

---

## 3) Define the 8th functionality precisely (operational definition)

### 3.1 Proposed definition for Candidate A

> **Learning/Memory:** A persistent internal mechanism that updates policy/state based on experience during an organism’s lifetime, improving task performance under changing or novel conditions, beyond what fixed genetics or fixed controllers achieve.

### 3.2 Extend your “functional analogy” rubric to the 8th

Reuse your three conditions (dynamic process, ablation effect, coupling), but add **one extra** requirement tailored to learning:

4. **Experience-dependence:** performance improves as a function of experience *within* an organism’s lifetime under stationary genetics.

That makes the 8th feel like a *scientific* criterion, not a feature.

---

## 4) Implementation design (keep it minimal but convincing)

You already have:

* NN controller (8→16→4) with genome-encoded weights
* internal state vector for homeostasis
* environment resource dynamics
* ablation toggles per criterion
* held-out seeds, strong stats pipeline 

### 4.1 Minimal viable implementation paths for Learning/Memory

Choose **one** (you can mention alternatives as future work).

#### Option A1: **Plastic synapses (online weight update)**

* Keep genome as “initial weights”
* Add a small online update rule for a subset of weights (e.g., last layer only)
* Update uses local signals: reward proxy, prediction error, energy delta, boundary change

**Pros:** very “life-like,” classic ALife/neuronal plasticity vibe
**Cons:** risk of instability; reviewers might ask about compute confounds

#### Option A2: **External or internal memory state (RNN-lite)**

* Add a memory vector `m_t` (e.g., 8–32 dims)
* The controller consumes `(obs_t, m_t)` and outputs `(action_t, m_{t+1})`
* Genome encodes the transition weights, but memory evolves during lifetime

**Pros:** stable, easy to ablate (zero memory), clean interpretation
**Cons:** some reviewers call it “just bigger controller” unless you emphasize experience dependence

#### Option A3: **Meta-learning-ish “fast variables”**

* Genome encodes both slow parameters and a fast adaptive mechanism
* Fast variables update from experience; slow variables evolve

**Pros:** beautifully bridges learning + evolution
**Cons:** more complex; may be too big for one cycle

**My recommendation:** **Option A2** is the cleanest for a conference full paper: stable, ablation-friendly, and easy to demonstrate experience-dependence.

---

## 5) Experiment suite (the heart of the “drastic” paper)

The key is: you must test the 8th on **tasks where the original 7 are insufficient**.

### 5.1 New evaluation paradigm: “Novel perturbation generalization”

Add environment regimes where fixed policies fail:

1. **Resource relocation shock**

   * resources move or shift hotspots mid-run
2. **Poison/waste hazard shift**

   * waste penalty dynamics change unexpectedly
3. **Sensor corruption / partial observability**

   * degrade sensing temporarily
4. **Adversarial ecology**

   * introduce parasite agents or predators (lightweight)

You already talk about mid-run ablation protocols and reproducibility constraints; this integrates perfectly. 

### 5.2 Conditions (must include strong controls)

For each perturbation regime, run:

1. **Baseline (7 criteria, no 8th)**
2. **+8th functionality enabled**
3. **8th ablated** (toggle learning/memory off, keep everything else same)
4. **Compute-matched sham control**

   * same compute cost, but memory updates are random or state-neutral
5. **Retuned baseline** (important!)

   * allow baseline to tune hyperparameters to match best possible fixed-controller performance
   * prevents criticism: “you just didn’t tune the original system enough”

### 5.3 Primary outcomes (choose 2–3, not 10)

Use metrics aligned with your current framework but targeting generalization:

* **Robustness under perturbation:** survival/population AUC after shock
* **Recovery time:** time to return to stable population band
* **Out-of-distribution score:** performance gap between train-regime and novel-regime

Secondary:

* individual lifespan (to separate individual vs population effects, as you already do) 
* spatial cohesion (if relevant) 

### 5.4 “Experience-dependence” demonstration (must-have figure)

Ablate **experience** while keeping the same memory mechanism:

* same organism genotype
* run two scenarios:

  1. **With prior exposure** to the perturbation pattern
  2. **Without exposure**
* show within-lifetime improvement (e.g., later episodes better than early episodes)

This is the figure that makes it “learning,” not “just larger state.”

---

## 6) Causality: show the 8th is not redundant with existing criteria

This is where papers become *drastically* stronger.

### 6.1 Variance decomposition / incremental explanatory power

Fit a simple model predicting robustness outcomes using the 7 criterion signals you already measure (energy, boundary integrity, internal state, etc.), then add “learning-active” indicators.

Show:

* **R² (or AUC/accuracy)** improves materially when adding 8th
* effect persists across held-out seeds

### 6.2 Non-reducibility test (“retuning can’t fake it”)

Let the baseline (no 8th) tune:

* sensing range
* movement policy weights (genetic)
* homeostasis decay rates
* reproduction thresholds 

If learning still wins under novel perturbations, the 8th looks real.

---

## 7) How this becomes an ALIFE paper (story + structure)

### Proposed title direction

* “Beyond the Seven Criteria: Learning and Memory as an Eighth Axis of Life-Likeness in Digital Organisms”
* or “Fast Adaptation as an Eighth Life Functionality: A Criterion-Ablation Test in a Hybrid Swarm-Organism System” 

### Paper outline (tight)

1. **Motivation:** seven criteria ≠ sufficient for robustness/generalization
2. **Framework:** extend functional analogy to 8th; define experience-dependence
3. **Implementation:** memory mechanism + clean ablation toggle
4. **Tasks:** perturbation generalization suite
5. **Results:** +8th improves robustness; ablation kills it; sham doesn’t help
6. **Analysis:** variance decomposition + non-reducibility test
7. **Implications:** astrobiology (fast adaptation), AI agents (life-like robustness), ALife theory

---

## 8) “Reviewer trap” checklist (things to preempt)

1. **“It’s just a bigger neural net.”**

   * Counter: show experience dependence + ablation + sham control + retuned baseline.

2. **“Compute confound.”**

   * Counter: compute-matched sham.

3. **“Not orthogonal; it’s just response/homeostasis.”**

   * Counter: novelty perturbations + incremental explanatory power + non-reducibility test.

4. **“Definitions of life are philosophical.”**

   * Counter: you keep the claim operational and falsifiable (your paper already does this well). 

---

## 9) Concrete deliverables (what you should aim to produce)

### Code/artifacts

* `enable_learning_memory` toggle
* memory state logging
* perturbation regime scripts
* sham-control implementation
* figure scripts + manifests (consistent with your reproducibility design) 

### Figures (suggested “must-have 4”)

1. **Perturbation performance** (baseline vs +8th vs 8th-ablated vs sham)
2. **Recovery curves** after shocks
3. **Experience-dependence** (performance improves within lifetime / across episodes)
4. **Variance decomposition** (7 criteria signals vs +8th)

---

## Archived: Pre-Experiment Questionnaire

*The following questions guided the initial experiment design (answered during Phase 0 planning). Preserved for reference.*

<details>
<summary>Click to expand</summary>

**Thesis (original):** *The textbook seven criteria are not sufficient to explain resilience and generalization under novel perturbations; an additional functionality—**Learning/Memory (within-lifetime adaptation)** / **Collective Organization** / **Novelty Generation**—accounts for systematic variance beyond the seven, and its necessity can be tested with the same falsifiable ablation framework you already established.*

1. Which 8th? → A) Learning/Memory (Phase 1), then B) Collective organization (Phase 3)
2. Implementation? → EMA-driven homeostatic correction (separate phase, not NN input expansion)
3. Perturbations? → Famine (resource drop at step 3000), Boom-bust (cyclic period 2500)
4. Framing? → “8th functionality axis” (safer for ALIFE)
5. Compute budget? → n=30 seeds × 10k steps per condition (4 conditions × 2 regimes)
6. External substrate? → No, single system only

</details>

---

## 10) Experimental Results (completed)

### Status: Both candidates tested, both null. Paper complete (`paper/main.tex`).

The following summarizes the completed experiments across Phases 1–3.

### Candidate A: Learning & Memory (within-lifetime adaptation)

**Implementation**: EMA-driven homeostatic correction phase. 2-element EMA tracking mean internal-state channels IS[0] and IS[1], with configurable decay $\alpha$. Four genome-encoded parameters (2 gains, 2 targets) modulate correction strength and set-point. NN architecture unchanged (212 weights for 8-input base topology). Sham: uniform random draws replace EMA trace each timestep at same compute cost.
- Source: `crates/life-criteria-core/src/world/phases/memory.rs`, `crates/life-criteria-core/src/genome.rs` (segment 7)

**Suite 1 — Normal conditions** (30 seeds × 10k steps, held-out seeds 100–129):
- Survival AUC: +106.5 vs baseline ($d = 0.13$, $p_\text{adj} = 0.94$) → **null**
- Memory mechanism verified: EMA late variance $p < 10^{-9}$ vs sham
- Artifacts: `experiments/criterion8_manifest.json`, `experiments/criterion8_analysis.json` (keys: `pairwise_vs_baseline.criterion8_on`, `memory_stability`)

**Suite 2 — Famine stress** (resource drop at step 3000):
- Post-shock AUC: +7.3 ($d = 0.03$, $p_\text{adj} = 1.0$) → **null**
- Extinction: 93.3% enabled vs 86.7% baseline
- Artifacts: `experiments/stress_manifest.json`, `experiments/stress_analysis.json` (key: `famine.pairwise_vs_baseline.criterion8_on`)

**Suite 3 — Boom-bust stress** (cyclic period 2500):
- Survival AUC: −103.2 ($d = -0.28$, $p_\text{adj} = 0.46$) → **null**
- Extinction: 93.3% enabled, 93.3% baseline
- Artifacts: `experiments/stress_manifest.json`, `experiments/stress_analysis.json` (key: `boom_bust.pairwise_vs_baseline.criterion8_on`)

**Diagnosis**: Memory mechanism converges correctly (EMA works) but provides no survival advantage. The perturbation regimes lack learnable temporal structure for memory to exploit.

### Candidate B: Collective Organization (kin-sensing)

**Implementation**: `kin_fraction` channel (input 8→9 dims, 212→228 weights). Single-pass `count_neighbors_split()` for kin/non-kin counting. Sham: permute real kin_fraction across alive agents.
- Source: `crates/life-criteria-core/src/spatial.rs` (`count_neighbors_split`), `crates/life-criteria-core/src/world/phases/nn_query.rs` (kin_fraction input), `crates/life-criteria-core/src/nn.rs` (INPUT_SIZE=9)

**Suite 4 — Famine stress**:
- Post-shock AUC: −38.3 ($d = -0.13$, $p_\text{adj} = 1.0$) → **null**
- Extinction: 96.7% enabled vs 86.7% baseline
- Artifacts: `experiments/candidateB_stress_manifest.json`, `experiments/candidateB_stress_analysis.json` (key: `famine.pairwise_vs_baseline.candidateB_on`; kin fraction: `famine.summaries.candidateB_on.kin_fraction_final`)

**Suite 5 — Boom-bust stress**:
- Survival AUC: −32.3 ($d = -0.08$, $p_\text{adj} = 1.0$) → **null**
- Extinction: 76.7% enabled vs 83.3% baseline
- Artifacts: `experiments/candidateB_stress_manifest.json`, `experiments/candidateB_stress_analysis.json` (key: `boom_bust.pairwise_vs_baseline.candidateB_on`; kin fraction: `boom_bust.summaries.candidateB_on.kin_fraction_final`)

**Diagnosis**: Organisms converge to ~1 agent each under population cap, making `kin_fraction` degenerate (mean 0.03 famine, 0.19 boom-bust—high variance driven by rare multi-agent survivors). This is an **observability failure**: the kin signal exists in principle but becomes sparse under the tested demographic regime.

### Summary

| Candidate | Regime | Δ AUC | Cohen’s d | 95% CI(d) | p_adj | Result |
|-----------|--------|-------|-----------|-----------|-------|--------|
| A (Memory) | Normal | +106.5 | 0.13 | [−0.38, 0.64] | 0.94 | Null |
| A (Memory) | Famine | +7.3 | 0.03 | [−0.48, 0.54] | 1.00 | Null |
| A (Memory) | Boom-bust | −103.2 | −0.28 | [−0.79, 0.23] | 0.46 | Null |
| B (Kin) | Famine | −38.3 | −0.13 | [−0.64, 0.38] | 1.00 | Null |
| B (Kin) | Boom-bust | −32.3 | −0.08 | [−0.59, 0.43] | 1.00 | Null |

**Point estimates vs SESOI.** All enabled-vs-baseline point estimates are small: $|d| \leq 0.28$ (largest: $d = -0.28$, boom-bust Candidate A), well below the pre-specified SESOI of $d = 0.5$.

**Confidence intervals and power.** The 95% CIs for Cohen's $d$ extend past $\pm 0.5$ for every comparison (typical half-width $\approx 0.51$), due to the sample size ($n = 30$ per condition). No single comparison's CI falls entirely within $[-0.5, 0.5]$.

**Interpretation.** Formal equivalence via TOST cannot be claimed. The results support a **bounded-null interpretation**: observed effects are consistently small, but the data cannot definitively exclude medium effects in either direction. Larger samples or meta-analytic combination across regimes would be needed for a formal equivalence claim.

### Pivot outcome

Per the pivot strategy, both Candidate A and B are null. The contribution pivots from “we found the 8th” to:
1. **Validated falsifiable protocol** for testing candidate criteria
2. **Bounded null results** with mechanistic diagnoses
3. **Design lessons** for future ALife systems testing experiential/collective criteria

Paper: `paper/main.tex` — “Searching for an Eighth Criterion of Life: A Falsifiable Framework and Two Null Results”

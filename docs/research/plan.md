# Research Plan (ALIFE Full Paper): Discovering an “8th Functionality” Beyond the Textbook 7

*(A detailed Markdown document you can iterate on)*

This plan assumes your current system already implements the **seven textbook criteria** as **interdependent, ablatable dynamic processes** with a strong experimental framework (held-out seeds, ablations, coupling evidence, proxy controls). 
The goal here is to make a **drastic conceptual jump**: not “we added another feature,” but **we discovered an additional axis that explains survival/adaptation/resilience beyond the seven**.

---

## 0) One-sentence thesis (what ALIFE reviewers should remember)

**Thesis:** *The textbook seven criteria are not sufficient to explain resilience and generalization under novel perturbations; an additional functionality—**Learning/Memory (within-lifetime adaptation)** / **Collective Organization** / **Novelty Generation**—accounts for systematic variance beyond the seven, and its necessity can be tested with the same falsifiable ablation framework you already established.* 

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

# Questions for you (so I can update this document precisely)

Please answer as many as you can; even short answers are fine.

1. **Which 8th are you leaning toward right now?**

   * A) Learning/Memory (within lifetime)
   * B) Collective organization
   * C) Open-ended novelty
   * D) Other (describe)

2. If **Learning/Memory**: which implementation feels most compatible with your current codebase?

   * A1 plastic weights, A2 memory vector (RNN-lite), A3 fast variables

3. **What perturbations are easiest to implement in your environment today?**

   * resource relocation, waste toxicity change, sensor corruption, predators/parasites, other?

4. Do you want the 8th to be framed as:

   * **(i)** “8th life criterion” (strong claim)
   * **(ii)** “8th functionality axis for life-likeness / robustness” (safer, often better for ALIFE)

5. What’s your **compute budget** for the new experiments (roughly):

   * same as current (n=30 seeds × 2000 steps), or can you do longer/more regimes?

6. Are you open to including **one external substrate** comparison (even a minimal one), or must this stay within your current system?

If you answer these, I’ll revise this Markdown into a more *execution-ready* version with: exact experimental grid, exact metrics, and a proposed “main theorem-like claim + falsification tests” tailored to your constraints.

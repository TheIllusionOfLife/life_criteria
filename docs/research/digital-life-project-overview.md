# Digital Life: Functional Analogy of All Seven Biological Criteria in Computational Organisms

> **Document Role:** Initial Project Proposition (1/3)
> This is the foundational document that defines the research vision, technical approach, and prototyping roadmap.
> - Reviewed in: [`unified-review.md`](unified-review.md) — identifies critical gaps, risks, and recommendations
> - Addressed in: [`action-plan.md`](action-plan.md) — concrete 8-week plan incorporating review feedback and researcher decisions

## Project Overview

**Core Question:** Can we build a computational system where autonomous digital entities satisfy all seven biological criteria for life — not as checkboxes, but through genuine functional analogy?

**What's Novel:** Existing ALife systems (Tierra, Avida, Polyworld, Lenia) satisfy individual criteria at varying depths but typically rely on simplified proxies — e.g., "energy points" for metabolism, pre-defined agent boundaries for cellular organization. No system has implemented all seven criteria as structurally equivalent computational processes. This project takes each criterion seriously as a design constraint and builds a unified system from scratch.

**Approach:** Environment/substrate or swarm-collective system using neural networks, agent-based modeling, and evolutionary computation.

**Success Criteria (Dual):**
1. **Measurable:** Quantitative metrics confirm each of the 7 criteria is functionally present
2. **Perceptual:** The system produces behavior that looks and feels alive to a human observer

**Goal:** Publication. Fast early prototyping to assess feasibility.

---

## The Seven Criteria: Technical Specification

### 1. Cellular Organization
**Biological reality:** A self-maintaining boundary (membrane) that separates inside from outside, with internal compartmentalization.

**Functional analog:** Each digital organism maintains a computational boundary that requires active energy expenditure. The boundary isn't a static data structure — it's a process. If the organism stops maintaining it, the boundary degrades and the organism dissolves into the environment.

**Technical approaches:**
- (A) Agent-based: Each agent has an active membrane process that costs metabolic resources to maintain. Internal state is shielded from direct environmental access.
- (C) Swarm: The "cell membrane" is the emergent boundary of a collective — maintained by active coordination of peripheral agents. If coordination breaks down, the group dissolves.

**Risk: 2.5/5 — Medium.** Self-maintaining boundaries are studied in autopoiesis literature. The swarm-boundary variant is more novel. Key question: can boundaries emerge rather than be pre-designed?

**Time to prototype: 2-4 weeks**

---

### 2. Metabolism
**Biological reality:** A network of interdependent chemical reactions that transform raw resources into energy, structural components, and waste products. The network itself is what keeps the organism alive.

**Functional analog:** The organism must process environmental resources through a multi-step transformation pipeline. Not "eat food → gain 10 energy points" but a network of computational processes:
- Raw resources (environmental data, compute tokens, spatial resources) enter through the boundary
- Multiple transformation steps convert raw inputs into usable forms
- Intermediate products feed into other pathways (interdependence)
- Waste products are expelled
- If any pathway breaks, the organism degrades over time

**Technical approaches:**
- Graph-based metabolic networks where nodes are transformation functions and edges are resource flows
- Genetically encoded: the genome specifies the metabolic network topology
- Evolved: metabolic networks can mutate and recombine across generations
- Resource types: could map to different computational resources (memory, cycles, information, spatial territory)

**Risk: 3.5/5 — Medium-High.** This is the criterion where most ALife systems cheat. Making metabolism genuinely network-like while keeping computation tractable is the core challenge. This is likely the most novel technical contribution of the project.

**Time to prototype: 3-6 weeks**

**Go/No-Go:** Can a metabolic network sustain an organism for >1000 timesteps without manual intervention? If yes, proceed. If the network always collapses or trivializes, rethink.

---

### 3. Homeostasis
**Biological reality:** Active regulation of internal state against external perturbation. Body temperature, pH, ion concentrations — all maintained within viable ranges through feedback loops.

**Functional analog:** The organism has internal state variables (e.g., metabolic throughput, boundary integrity, internal resource levels) that must stay within viable ranges. A neural controller monitors these states and adjusts behavior to maintain them:
- If resource levels drop → forage more aggressively
- If boundary is damaged → redirect resources to repair
- If internal temperature (metaphor for processing rate) drifts → compensate

The environment should periodically perturb these states (resource scarcity, environmental shocks) to test whether homeostasis is genuinely active.

**Technical approaches:**
- Recurrent neural network controllers that learn homeostatic regulation
- Could be evolved (genetic encoding of controller weights) or learned (lifetime adaptation) or both
- Internal state represented as a vector; viable range defined per dimension
- Death occurs when any state variable leaves its viable range for too long

**Risk: 2.5/5 — Medium.** Neural controllers for regulation are well-studied. The question is whether homeostatic behavior can emerge through evolution rather than being hand-designed. Yuya's NN expertise is directly applicable.

**Time to prototype: 2-4 weeks**

---

### 4. Growth and Development
**Biological reality:** Organisms begin simple and become more complex over their lifespan, following a developmental program encoded in their genome.

**Functional analog:** Each organism starts as a minimal "seed" — perhaps a genome and a bootstrap metabolic network — and develops over its lifetime:
- Metabolic network expands (new pathways added based on developmental program)
- Boundary grows in size/complexity
- Sensory/motor capabilities develop
- Neural controller architecture grows (adding neurons/connections over time)

The developmental program is genetically encoded and heritable, meaning that development itself evolves across generations.

**Technical approaches:**
- NCA-inspired developmental programs: genome encodes local rules for growth
- Staged development: distinct phases (juvenile → adult) with different capabilities
- Growth requires metabolic resources (connecting development to metabolism)
- In swarm context: group starts small and recruits or produces new members

**Risk: 2.5/5 — Medium.** NCA developmental models are well-studied. The challenge is coupling development tightly with metabolism (growth consumes resources) and making it genetically encoded and evolvable.

**Time to prototype: 2-4 weeks**

---

### 5. Reproduction
**Biological reality:** The organism produces offspring that carry genetic information with variation. The offspring are new entities, not extensions of the parent.

**Functional analog:** The organism itself initiates reproduction — not the system. When an organism has accumulated sufficient metabolic resources and reached developmental maturity:
- It divides (agent splits into two, each with copy of genome + mutations)
- Or it buds (swarm pinches off a sub-group with a subset of the collective genome)
- Offspring start as "seeds" and undergo their own development
- Genome includes mutation/variation mechanisms (point mutations, crossover if sexual reproduction)

**Technical approaches:**
- Asexual: organism copies genome with mutation, splits resources with offspring
- Sexual: two organisms exchange genetic material, produce offspring with recombined genome
- Swarm budding: collective reaches critical mass and divides
- Reproduction is costly (metabolic drain) — creating selection pressure for optimal timing

**Risk: 1.5/5 — Low.** Self-replication with variation is the most well-studied aspect of ALife. The novelty here is in coupling it with genuine metabolism (reproduction costs resources) and development (offspring must develop, not appear fully formed).

**Time to prototype: 1-2 weeks**

---

### 6. Response to Stimuli
**Biological reality:** Organisms sense their environment and change behavior accordingly — a plant turns toward light, an animal flees a predator.

**Functional analog:** Digital organisms have sensory inputs from their computational environment:
- Detect resource gradients (where is food?)
- Sense proximity of other organisms (friend or threat?)
- Perceive environmental conditions (resource scarcity, crowding, hazards)
- Process sensory information through neural controller
- Produce motor outputs (movement, foraging behavior, defensive posture, mating signals)

**Technical approaches:**
- Sensory field around each organism (local perception, not global knowledge)
- Neural network processing (feedforward or recurrent, evolved architecture)
- Behavioral repertoire emerges from evolution, not pre-scripted
- In swarm context: individual agents sense locally; collective behavior emerges

**Risk: 1.5/5 — Low.** Standard agent-based systems with neural controllers handle this well. Yuya's expertise in neural networks and agent systems is directly applicable.

**Time to prototype: 1-2 weeks**

---

### 7. Evolution / Adaptation
**Biological reality:** Populations change over generations through heritable variation and natural selection. Over deep time, this produces increasing complexity and novel adaptations.

**Functional analog:** Two levels:

**Level A — Basic Evolution (well-studied):**
- Heritable genomes encoding metabolism, development, neural controller, behavior
- Variation through mutation and recombination
- Differential survival based on metabolic efficiency, environmental fitness
- Adaptation to changing environments

**Level B — Open-Ended Evolution (frontier):**
- Continuous generation of genuine novelty (not just parameter tuning)
- Emergence of new organizational levels (single → colonial → multicellular-like)
- Evolution of evolvability itself (the evolutionary mechanism can change)
- No convergence to a fixed attractor

**Technical approaches:**
- Genome: variable-length encoding of metabolic network + developmental program + neural architecture
- Genetic operators: point mutation, insertion/deletion, duplication, crossover
- Environmental dynamics: changing conditions prevent stagnation
- Complexity metrics: track whether populations are generating genuine novelty over time

**Risk: 2/5 for Level A, 4.5/5 for Level B.** Basic evolutionary dynamics are straightforward. Open-ended evolution is the unsolved frontier of the entire ALife field.

**Recommendation:** Target Level A for the paper. Frame Level B as a long-term research direction and evaluate whether the system shows any signs of open-endedness.

**Time to prototype: Level A: 2-4 weeks. Level B: ongoing open question.**

---

## Risk Assessment Summary

| Criterion | Risk (1-5) | Time to Prototype | Notes |
|-----------|:----------:|:------------------:|-------|
| 1. Cellular Organization | 2.5 | 2-4 weeks | Swarm boundary is the novel angle |
| 2. Metabolism | **3.5** | **3-6 weeks** | **Highest risk. Most novel. Test first.** |
| 3. Homeostasis | 2.5 | 2-4 weeks | NN controllers; can it emerge? |
| 4. Growth/Development | 2.5 | 2-4 weeks | NCA-inspired; couple with metabolism |
| 5. Reproduction | 1.5 | 1-2 weeks | Well-studied; novelty in coupling |
| 6. Response to Stimuli | 1.5 | 1-2 weeks | Standard agent-based approach |
| 7. Evolution (basic) | 2.0 | 2-4 weeks | Well-studied |
| 7b. Open-Ended Evolution | 4.5 | Ongoing | Stretch goal, not required for paper |
| **Integration of all 7** | **3.0** | **4-8 weeks** | **Interaction effects are unknown** |

**Overall Risk Profile: YELLOW — Feasible with focused effort, but metabolism is the make-or-break.**

**Red Flags:** None currently. No risk-5 items required for the paper. Metabolism (3.5) is testable early.

**Green Lights:**
- Highest-risk item (metabolism) is testable in first 3-6 weeks
- Multiple criteria are low-risk and well-studied
- Your technical skills (NN, agents, evo comp) map directly to the requirements
- Swarm angle provides novelty even if individual criteria aren't new

---

## Parameter Strategy

**Fixed Parameters (the constraints that define this project):**
1. **All 7 criteria must be present** — this IS the project
2. **Functional analogy, not checkboxes** — each criterion is a genuine computational process
3. **Computational substrate** — this is digital, not wetware
4. **Autonomous operation** — no human intervention during runtime

**Floating Parameters (creative freedom):**
- Specific architecture (agent-based vs. NCA vs. hybrid vs. swarm)
- Programming language / framework
- Scale (number of organisms, environment size)
- Visualization approach
- Whether individual or swarm is primary
- Dimensionality (2D vs. 3D)
- Specific resource types

**Your Competitive Advantage:**
- Neural networks + agent systems + evolutionary computation (rare combination for ALife)
- Game development experience (visualization, real-time simulation, "feels alive")
- AI research background (rigorous evaluation frameworks)
- Bilingual (Japanese + English) — relevant for engaging Japanese ALife community

---

## Prototyping Roadmap

### Phase 0: Literature Deep-Dive (Week 1)
- Read ALIFE 2025 Kyoto proceedings for most recent work
- Study Polyworld source code (GitHub) for integration lessons
- Review Flow-Lenia and Coralai for continuous-substrate approaches
- **Deliverable:** Annotated bibliography of 10-15 key papers

### Phase 1: Metabolism + Cellular Organization (Weeks 2-5) ← HIGHEST RISK, TEST FIRST
- Build the metabolic network substrate
- Implement active boundary maintenance
- Test: Can a single organism sustain itself for >1000 timesteps?
- **Go/No-Go:** If metabolic network can't sustain, pivot approach before investing more

### Phase 2: Homeostasis + Stimuli Response (Weeks 6-8)
- Add internal state regulation (neural controller)
- Add sensory inputs and motor outputs
- Environmental perturbations to test homeostasis
- **Test:** Does the organism recover from perturbation?

### Phase 3: Growth/Development + Reproduction (Weeks 9-11)
- Implement developmental programs (genome → organism)
- Add reproduction with metabolic cost
- Test: Can organisms reproduce viable offspring that develop correctly?

### Phase 4: Evolution (Weeks 12-14)
- Population dynamics (multiple organisms, resource competition)
- Heritable variation and natural selection
- Test: Does the population adapt to environmental changes?

### Phase 5: Integration + Evaluation (Weeks 15-18)
- All 7 criteria running simultaneously
- Quantitative evaluation framework (metrics for each criterion)
- "Feels alive" subjective evaluation
- Visualization polish
- **Dual evaluation:** Do the metrics confirm life? Does it look alive?

### Phase 6: Paper Writing (Weeks 19-22)
- Write up results
- Create figures and demonstrations
- Submit to target venue

**Total estimated timeline: ~5.5 months**

---

## Evaluation Framework

### Quantitative Metrics (per criterion)

| Criterion | Metric | Threshold for "Alive" |
|-----------|--------|----------------------|
| Cellular Org. | Boundary integrity over time; ratio of maintained vs. degraded boundaries | >80% of organisms maintain boundary throughout lifespan |
| Metabolism | Metabolic network throughput; pathway interdependence measure | Network has >3 interdependent pathways; organism dies if any pathway is severed |
| Homeostasis | Recovery time after perturbation; variance of internal states | Internal states return to viable range within N timesteps after perturbation |
| Growth/Dev. | Complexity increase over lifespan (component count, network size) | Organisms at maturity are >2x complex vs. at birth |
| Reproduction | Offspring viability rate; genetic diversity in population | >50% of offspring survive to reproductive age |
| Stimuli | Behavioral response entropy; stimulus-response correlation | Behavior changes significantly in response to environmental changes |
| Evolution | Fitness increase over generations; trait diversity | Population mean fitness increases over 100+ generations |

### Perceptual Evaluation
- Record simulation videos
- Blind evaluation: show to observers, ask "does this look alive?"
- Compare against controls (random motion, pre-scripted behavior)
- Animacy perception metrics from cognitive science literature

### Comparison to Existing Systems
- Run same metrics on Polyworld, Lenia, Avida (where possible)
- Show that this system scores higher on metabolism and homeostasis criteria specifically

---

## Target Venues

### Primary Target: ALIFE 2026
- **Location:** Waterloo, Ontario, Canada
- **Dates:** August 17-21, 2026
- **Paper Deadline:** April 6, 2026 (~8 weeks from now — tight for full system, but feasible for Phase 1-2 results + framework paper)
- **Theme:** "Living and Lifelike Complex Adaptive Systems" — perfect fit
- **Note:** Joint with AMMCS 2026 (Applied Mathematics)

### Alternative Targets:
- **Artificial Life journal (MIT Press):** No deadline pressure; can submit when ready. Higher bar but more thorough paper.
- **GECCO 2026:** July 13-17, San José, Costa Rica. Deadline passed for this year.
- **ALIFE 2027:** If timeline doesn't work for 2026.

### Stretch Targets (if results are strong):
- Nature Machine Intelligence
- PNAS (as a scientific contribution on defining digital life)
- Frontiers in Robotics and AI

---

## Timeline Options

### Option A: Target ALIFE 2026 (Aggressive)
- **Now → April 6:** 8 weeks. Build Phase 1-2, write framework paper with preliminary results.
- **Contribution:** Novel framework for functional-analogy ALife + metabolism/homeostasis prototype
- **Risk:** Tight timeline; may only have partial system
- **Pro:** Gets work out quickly; ALIFE community is the right audience

### Option B: Target Artificial Life Journal (Thorough)
- **Now → August 2026:** ~6 months. Complete system through Phase 5.
- **Contribution:** Full system with all 7 criteria + comprehensive evaluation
- **Risk:** Longer commitment; need sustained motivation
- **Pro:** More complete story; higher impact per paper

### Option C: Two-Paper Strategy (Recommended)
- **Paper 1 (ALIFE 2026):** Framework + metabolism/homeostasis prototype (submit April)
- **Paper 2 (Journal, late 2026):** Complete system with all 7 criteria + evaluation
- **Pro:** Early publication builds credibility; full paper benefits from conference feedback
- **Risk:** More total work; but de-risks by getting partial results published

---

## Key Decision Points

| When | Decision | Go Criteria | Pivot Criteria |
|------|----------|-------------|----------------|
| Week 5 | Metabolism works? | Network sustains organism >1000 steps | Simplify metabolic model or change substrate |
| Week 8 | Homeostasis emerges? | Neural controller learns regulation | Hand-design initial controllers, evolve later |
| Week 14 | System integrates? | All 7 criteria function simultaneously | Focus paper on the criteria that work; frame gaps as future work |
| Week 18 | "Feels alive"? | Observers identify system as "alive" above chance | Improve visualization; may indicate criteria aren't deep enough |

---

## Open Questions for Next Session

1. **Agent-based or swarm-first?** We should pick one to start prototyping. Which excites you more?
2. **Programming language/framework?** Python (fast prototyping) vs. Rust/C++ (performance) vs. game engine (Unity/Godot for visualization)?
3. **Environment:** 2D grid? Continuous 2D space? 3D? This affects visualization and computational cost.
4. **Metabolism specifics:** What should the "resources" be? Abstract tokens? Information? Spatial territory?

---

## References

**Foundational:**
- Fischbach & Walsh (2024). "Problem choice and decision trees in science and engineering." Cell, 187, 1828-1833.
- Langton (1989). "Artificial Life." Proceedings of the First Workshop on the Synthesis and Simulation of Living Systems.

**Key Systems:**
- Ray (1991). Tierra: self-replicating programs in virtual memory
- Adami & Ofria (2003). Avida: digital evolution platform
- Chan (2019). Lenia: biology of artificial life
- Yaeger (1994). Polyworld: life in a new context
- Flow-Lenia (2025). Emergent evolutionary dynamics in mass-conservative systems
- Coralai, ENIGMA, Biomaker CA (ALIFE 2024). Neural cellular automata ecosystems

**Open-Ended Evolution:**
- Banzhaf et al. (2016). "Defining and simulating open-ended novelty."
- Packard et al. (2019). "An overview of open-ended evolution."

**Autopoiesis:**
- Maturana & Varela (1980). Autopoiesis and Cognition.
- Beer (2004). "Autopoiesis and cognition in the game of life."

**Animacy Perception:**
- Heider & Simmel (1944). Apparent behavior study.
- Tremoulet & Feldman (2000). "Perception of animacy from the motion of a single object."

---

*Document generated: February 8, 2026*
*Framework: Fischbach & Walsh Scientific Problem Selection*

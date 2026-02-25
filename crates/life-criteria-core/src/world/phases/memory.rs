use rand::Rng;

use super::super::World;

impl World {
    /// Update per-organism memory traces and apply memory-driven homeostatic corrections.
    ///
    /// This is the 8th criterion: within-lifetime learning/memory.
    /// Each organism maintains a 2-element exponential moving average (EMA) of
    /// its agents' mean internal-state channels IS[0] and IS[1].  The EMA is used
    /// to compute a proportional correction that drives IS toward `memory_target`,
    /// modulated by the genome's memory segment (segment 7).
    ///
    /// Three-pass layout to satisfy Rust borrow rules:
    ///   1. Collect IS sums from agents (shared borrow of agents + organisms)
    ///   2. Pre-compute sham memory values if needed (mutable borrow of rng only)
    ///   3. Update `org.memory` and compute per-org corrections (exclusive borrow of organisms)
    ///   4. Apply corrections to agents (exclusive borrow of agents, shared borrow of corrections)
    ///
    /// **Sham mode** (`enable_sham_process = true`):
    /// The memory trace is replaced by a fresh `U(0,1)` draw each step instead of
    /// the EMA.  This gives compute-matched random corrections — same code path and
    /// same correction magnitude, but no temporal coherence (no learning).  Used as
    /// the control condition in Phase 1b experiments.
    ///
    /// When `enable_memory = false` this function returns immediately, so the call
    /// site in `World::step` has zero overhead in the baseline condition.
    pub(in crate::world) fn step_memory_phase(&mut self) {
        if !self.config.enable_memory {
            return;
        }

        let n_orgs = self.organisms.len();
        let decay = self.config.memory_decay;
        let base_gain = self.config.memory_gain;
        let base_target = self.config.memory_target;
        let dt = self.config.dt as f32;
        let is_sham = self.config.enable_sham_process;

        // -------------------------------------------------------------------
        // Pass 1: accumulate IS[0] and IS[1] sums per organism from agents
        // -------------------------------------------------------------------
        let mut is_sums = vec![[0.0f32; 2]; n_orgs];
        let mut is_counts = vec![0usize; n_orgs];

        for agent in &self.agents {
            let org_idx = agent.organism_id as usize;
            if !self.organisms.get(org_idx).is_some_and(|o| o.alive) {
                continue;
            }
            is_sums[org_idx][0] += agent.internal_state[0];
            is_sums[org_idx][1] += agent.internal_state[1];
            is_counts[org_idx] += 1;
        }

        // -------------------------------------------------------------------
        // Pass 2 (sham only): fill the reusable sham_vals_buffer.
        // Using self.sham_rng (not self.rng) keeps sham draws independent of the
        // main simulation stream.  Reusing the pre-allocated buffer avoids per-step
        // heap allocation.
        //
        // Note: the sham condition still runs ~1.7× slower than the EMA condition.
        // The overhead is not from these RNG draws but from a behavioral cascade:
        // random IS corrections destabilize organisms → more deaths/births → more
        // work in reproduction, mutation, and spatial phases per step.  This is
        // scientifically expected (sham is a harder environment), and does not
        // affect the validity of the comparison.
        if is_sham {
            self.sham_vals_buffer.resize(n_orgs, [0.0; 2]);
            for slot in &mut self.sham_vals_buffer {
                slot[0] = self.sham_rng.random::<f32>();
                slot[1] = self.sham_rng.random::<f32>();
            }
        }

        // -------------------------------------------------------------------
        // Pass 3: update EMA memory trace (or sham override) and compute corrections
        // -------------------------------------------------------------------
        let mut org_corrections = vec![[0.0f32; 2]; n_orgs];

        for (org_idx, org) in self.organisms.iter_mut().enumerate() {
            if !org.alive {
                continue;
            }
            let count = is_counts[org_idx].max(1) as f32;
            let mw = org.genome.memory_weights();

            for i in 0..2 {
                if is_sham {
                    // Sham: replace EMA with pre-computed random draw — no temporal learning
                    org.memory[i] = self.sham_vals_buffer[org_idx][i];
                } else {
                    // Real EMA: m = decay * m + (1 - decay) * mean_is
                    let mean_is = is_sums[org_idx][i] / count;
                    org.memory[i] = decay * org.memory[i] + (1.0 - decay) * mean_is;
                }

                // Genome-modulated gain and target (signed additive offsets).
                // Gains are clamped to ≥0; targets are clamped to [0,1].
                let gene_gain = mw.get(i).copied().unwrap_or(0.0);
                let gene_target = mw.get(i + 2).copied().unwrap_or(0.0);
                let eff_gain = (base_gain + gene_gain).max(0.0);
                let eff_target = (base_target + gene_target).clamp(0.0, 1.0);

                // Proportional correction toward target, scaled by dt
                org_corrections[org_idx][i] = eff_gain * (eff_target - org.memory[i]) * dt;
            }
        }

        // -------------------------------------------------------------------
        // Pass 4: apply memory corrections to agent internal states
        // -------------------------------------------------------------------
        for agent in &mut self.agents {
            let org_idx = agent.organism_id as usize;
            if !self.organisms.get(org_idx).is_some_and(|o| o.alive) {
                continue;
            }
            let corr = &org_corrections[org_idx];
            agent.internal_state[0] = (agent.internal_state[0] + corr[0]).clamp(0.0, 1.0);
            agent.internal_state[1] = (agent.internal_state[1] + corr[1]).clamp(0.0, 1.0);
        }
    }
}

use crate::spatial;

use super::super::World;

impl World {
    /// Compute neighbor-informed neural deltas for all agents.
    pub(in crate::world) fn step_nn_query_phase(&mut self, grid: &spatial::UniformGrid) {
        let deltas = &mut self.deltas_buffer;
        let neighbor_sums = &mut self.neighbor_sums_buffer;
        let neighbor_counts = &mut self.neighbor_counts_buffer;
        let kin_fracs = &mut self.agent_kin_fractions_buffer;
        let total_counts = &mut self.agent_total_counts_buffer;
        let agents = &self.agents;
        let organisms = &self.organisms;
        let config = &self.config;

        deltas.clear();
        deltas.reserve(agents.len());

        let org_count = organisms.len();
        if neighbor_sums.len() != org_count {
            neighbor_sums.resize(org_count, 0.0);
            neighbor_counts.resize(org_count, 0);
        }
        neighbor_sums.fill(0.0);
        neighbor_counts.fill(0);

        let use_kin_sensing = config.enable_collective_sensing || config.enable_sham_collective;

        // ------------------------------------------------------------------
        // Pass 1: compute per-agent kin_fraction and neighbor counts.
        // When collective sensing is disabled, we still need neighbor counts
        // for the existing channel 7 input.
        // Also accumulates encounter metrics (nc_sum, with_neighbors, alive_agents)
        // in the same pass to avoid a redundant iteration.
        // ------------------------------------------------------------------
        kin_fracs.clear();
        kin_fracs.reserve(agents.len());
        total_counts.clear();
        total_counts.reserve(agents.len());

        // Build agent ID → organism_id map for kin-sensing lookups.
        // Agent IDs are not contiguous after pruning, so we cannot use them
        // as slice indices.
        let agent_id_to_org: std::collections::HashMap<u32, u16> = if use_kin_sensing {
            agents.iter().map(|a| (a.id, a.organism_id)).collect()
        } else {
            std::collections::HashMap::new()
        };

        let mut nc_sum = 0.0f32;
        let mut with_neighbors = 0usize;
        let mut alive_agents = 0usize;

        for agent in agents {
            let org_idx = agent.organism_id as usize;
            if !organisms.get(org_idx).map(|o| o.alive).unwrap_or(false) {
                kin_fracs.push(0.0);
                total_counts.push(0);
                continue;
            }

            // Inline effective_sensing_radius logic to avoid borrow conflicts
            let dev_sensing = if config.enable_growth {
                organisms[org_idx]
                    .developmental_program
                    .stage_factors(organisms[org_idx].maturity)
                    .1
            } else {
                1.0
            };
            let effective_radius = config.sensing_radius * dev_sensing as f64;

            let (kf, tc) = if use_kin_sensing {
                let (kin_count, non_kin_count) = spatial::count_neighbors_split(
                    grid,
                    agent.position,
                    effective_radius,
                    agent.id,
                    agent.organism_id,
                    &agent_id_to_org,
                    config.world_size,
                );
                let total_count = kin_count + non_kin_count;
                let kin_fraction = if total_count > 0 {
                    kin_count as f32 / total_count as f32
                } else {
                    0.0
                };
                (kin_fraction, total_count)
            } else {
                let neighbor_count = spatial::count_neighbors(
                    grid,
                    agent.position,
                    effective_radius,
                    agent.id,
                    config.world_size,
                );
                (0.0, neighbor_count)
            };

            kin_fracs.push(kf);
            total_counts.push(tc);
            neighbor_sums[org_idx] += tc as f32;
            neighbor_counts[org_idx] += 1;

            // Accumulate encounter metrics in the same pass
            alive_agents += 1;
            nc_sum += tc as f32;
            if tc > 0 {
                with_neighbors += 1;
            }
        }

        // ------------------------------------------------------------------
        // Pass 1.5 (sham only): permute kin_fraction values across alive agents.
        // This preserves the empirical distribution of kin_fraction while breaking
        // the correlation between each agent's actual kin ratio and its NN input.
        // Uses sham_rng to avoid perturbing the main RNG stream.
        // ------------------------------------------------------------------
        if config.enable_sham_collective {
            use rand::seq::SliceRandom;
            // Collect indices of alive agents
            let alive_indices: Vec<usize> = agents
                .iter()
                .enumerate()
                .filter(|(_, a)| {
                    organisms
                        .get(a.organism_id as usize)
                        .map(|o| o.alive)
                        .unwrap_or(false)
                })
                .map(|(i, _)| i)
                .collect();

            // Extract their kin_fractions, shuffle, write back
            let mut alive_fracs: Vec<f32> = alive_indices.iter().map(|&i| kin_fracs[i]).collect();
            alive_fracs.shuffle(&mut self.sham_rng);
            for (&idx, &frac) in alive_indices.iter().zip(alive_fracs.iter()) {
                kin_fracs[idx] = frac;
            }
        }

        // ------------------------------------------------------------------
        // Finalize encounter metrics (kf_sum must come after sham permutation).
        // ------------------------------------------------------------------
        {
            let mut kf_sum = 0.0f32;
            for (i, agent) in agents.iter().enumerate() {
                let org_idx = agent.organism_id as usize;
                if organisms.get(org_idx).map(|o| o.alive).unwrap_or(false) {
                    kf_sum += kin_fracs[i];
                }
            }
            self.last_kin_fraction_sum = kf_sum;
            self.last_agents_with_neighbors = with_neighbors;
            self.last_neighbor_count_sum = nc_sum;
            self.last_alive_agent_count = alive_agents;
        }

        // ------------------------------------------------------------------
        // Pass 2: compute NN deltas using the 9-element input array.
        // ------------------------------------------------------------------
        for (agent_idx, agent) in agents.iter().enumerate() {
            let org_idx = agent.organism_id as usize;
            if !organisms.get(org_idx).map(|o| o.alive).unwrap_or(false) {
                deltas.push([0.0; 4]);
                continue;
            }

            let total_count = total_counts[agent_idx];
            let kin_fraction = if config.enable_collective_sensing || config.enable_sham_collective
            {
                kin_fracs[agent_idx]
            } else {
                0.0
            };

            let input: [f32; 9] = [
                (agent.position[0] / config.world_size) as f32,
                (agent.position[1] / config.world_size) as f32,
                (agent.velocity[0] / config.max_speed) as f32,
                (agent.velocity[1] / config.max_speed) as f32,
                agent.internal_state[0],
                agent.internal_state[1],
                agent.internal_state[2],
                total_count as f32 / config.neighbor_norm as f32,
                kin_fraction,
            ];
            let nn = &organisms[org_idx].nn;
            deltas.push(nn.forward(&input));
        }
    }
}

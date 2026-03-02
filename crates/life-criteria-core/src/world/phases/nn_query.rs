use crate::spatial;

use super::super::World;

impl World {
    /// Compute neighbor-informed neural deltas for all agents.
    pub(in crate::world) fn step_nn_query_phase(&mut self, grid: &spatial::UniformGrid) {
        let deltas = &mut self.deltas_buffer;
        let neighbor_sums = &mut self.neighbor_sums_buffer;
        let neighbor_counts = &mut self.neighbor_counts_buffer;
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
        // ------------------------------------------------------------------
        let mut agent_kin_fractions: Vec<f32> = Vec::with_capacity(agents.len());
        let mut agent_total_counts: Vec<usize> = Vec::with_capacity(agents.len());

        for agent in agents {
            let org_idx = agent.organism_id as usize;
            if !organisms.get(org_idx).map(|o| o.alive).unwrap_or(false) {
                agent_kin_fractions.push(0.0);
                agent_total_counts.push(0);
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

            if use_kin_sensing {
                let (kin_count, non_kin_count) = spatial::count_neighbors_split(
                    grid,
                    agent.position,
                    effective_radius,
                    agent.id,
                    agent.organism_id,
                    agents,
                    config.world_size,
                );
                let total_count = kin_count + non_kin_count;
                let kin_fraction = if total_count > 0 {
                    kin_count as f32 / total_count as f32
                } else {
                    0.0
                };
                agent_kin_fractions.push(kin_fraction);
                agent_total_counts.push(total_count);
                neighbor_sums[org_idx] += total_count as f32;
                neighbor_counts[org_idx] += 1;
            } else {
                let neighbor_count = spatial::count_neighbors(
                    grid,
                    agent.position,
                    effective_radius,
                    agent.id,
                    config.world_size,
                );
                agent_kin_fractions.push(0.0);
                agent_total_counts.push(neighbor_count);
                neighbor_sums[org_idx] += neighbor_count as f32;
                neighbor_counts[org_idx] += 1;
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
            let mut alive_fracs: Vec<f32> =
                alive_indices.iter().map(|&i| agent_kin_fractions[i]).collect();
            alive_fracs.shuffle(&mut self.sham_rng);
            for (&idx, &frac) in alive_indices.iter().zip(alive_fracs.iter()) {
                agent_kin_fractions[idx] = frac;
            }
        }

        // ------------------------------------------------------------------
        // Accumulate encounter metrics for StepMetrics.
        // ------------------------------------------------------------------
        {
            let mut kf_sum = 0.0f32;
            let mut with_neighbors = 0usize;
            let mut nc_sum = 0.0f32;
            let mut alive_agents = 0usize;
            for (i, agent) in agents.iter().enumerate() {
                let org_idx = agent.organism_id as usize;
                if !organisms.get(org_idx).map(|o| o.alive).unwrap_or(false) {
                    continue;
                }
                alive_agents += 1;
                let tc = agent_total_counts[i];
                nc_sum += tc as f32;
                if tc > 0 {
                    with_neighbors += 1;
                }
                kf_sum += agent_kin_fractions[i];
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

            let total_count = agent_total_counts[agent_idx];
            let kin_fraction = if config.enable_collective_sensing || config.enable_sham_collective
            {
                agent_kin_fractions[agent_idx]
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

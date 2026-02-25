use crate::spatial;
use crate::spatial::AgentLocation;
use rstar::RTree;

use super::super::World;

impl World {
    /// Compute neighbor-informed neural deltas for all agents.
    pub(in crate::world) fn step_nn_query_phase(&mut self, tree: &RTree<AgentLocation>) {
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

        for agent in agents {
            let org_idx = agent.organism_id as usize;
            // Manual lookup to avoid borrowing self methods
            if !organisms.get(org_idx).map(|o| o.alive).unwrap_or(false) {
                deltas.push([0.0; 4]);
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

            let neighbor_count = spatial::count_neighbors(
                tree,
                agent.position,
                effective_radius,
                agent.id,
                config.world_size,
            );

            neighbor_sums[org_idx] += neighbor_count as f32;
            neighbor_counts[org_idx] += 1;

            let input: [f32; 8] = [
                (agent.position[0] / config.world_size) as f32,
                (agent.position[1] / config.world_size) as f32,
                (agent.velocity[0] / config.max_speed) as f32,
                (agent.velocity[1] / config.max_speed) as f32,
                agent.internal_state[0],
                agent.internal_state[1],
                agent.internal_state[2],
                neighbor_count as f32 / config.neighbor_norm as f32,
            ];
            let nn = &organisms[org_idx].nn;
            deltas.push(nn.forward(&input));
        }
    }
}

use super::super::World;
use crate::spatial;
use crate::spatial::AgentLocation;
use rstar::RTree;

impl World {
    /// Apply optional sham work and environment updates.
    pub(in crate::world) fn step_environment_phase(&mut self, tree: &RTree<AgentLocation>) {
        if self.config.enable_sham_process {
            let mut _sham_sum: f64 = 0.0;
            for agent in &self.agents {
                let org_idx = agent.organism_id as usize;
                if !self.organisms.get(org_idx).is_some_and(|o| o.alive) {
                    continue;
                }
                let effective_radius = self.effective_sensing_radius(org_idx);
                let neighbor_count = spatial::count_neighbors(
                    tree,
                    agent.position,
                    effective_radius,
                    agent.id,
                    self.config.world_size,
                );
                _sham_sum += neighbor_count as f64;
            }
        }

        if self.config.environment_shift_step > 0
            && self.step_index == self.config.environment_shift_step
        {
            self.current_resource_rate = self.config.environment_shift_resource_rate;
        }

        if self.config.environment_cycle_period > 0 {
            let phase = (self.step_index / self.config.environment_cycle_period) % 2;
            self.current_resource_rate = if phase == 0 {
                self.config.resource_regeneration_rate
            } else {
                self.config.environment_cycle_low_rate
            };
        }

        if self.current_resource_rate > 0.0 {
            self.resource_field
                .regenerate(self.current_resource_rate * self.config.dt as f32);
        }
    }
}

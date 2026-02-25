use super::super::World;

impl World {
    /// Update age, growth stage, and crowding effects, then mark deaths.
    pub(in crate::world) fn step_growth_and_crowding_phase(
        &mut self,
        boundary_terminal_threshold: f32,
    ) {
        let mut to_kill = Vec::new();
        {
            let config = &self.config;
            let neighbor_sums = &self.neighbor_sums_buffer;
            let neighbor_counts = &self.neighbor_counts_buffer;

            for (org_idx, org) in self.organisms.iter_mut().enumerate() {
                if !org.alive {
                    continue;
                }
                org.age_steps = org.age_steps.saturating_add(1);
                if org.age_steps > config.max_organism_age_steps {
                    to_kill.push(org_idx);
                    continue;
                }

                if config.enable_growth && org.maturity < 1.0 {
                    let base_rate = 1.0 / config.growth_maturation_steps as f32;
                    let rate = base_rate * org.developmental_program.maturation_rate_modifier;
                    org.maturity = (org.maturity + rate).min(1.0);
                }

                let avg_neighbors = if neighbor_counts[org_idx] > 0 {
                    neighbor_sums[org_idx] / neighbor_counts[org_idx] as f32
                } else {
                    0.0
                };
                if avg_neighbors > config.crowding_neighbor_threshold {
                    let excess = avg_neighbors - config.crowding_neighbor_threshold;
                    org.boundary_integrity = (org.boundary_integrity
                        - excess * config.crowding_boundary_decay * config.dt as f32)
                        .clamp(0.0, 1.0);
                }
                if org.boundary_integrity <= boundary_terminal_threshold {
                    to_kill.push(org_idx);
                }
            }
        }

        for org_idx in to_kill {
            self.mark_dead(org_idx);
        }
    }
}

use super::super::World;

impl World {
    /// Update per-organism metabolism and consume resource field.
    pub(in crate::world) fn step_metabolism_phase(&mut self, boundary_terminal_threshold: f32) {
        if !self.config.enable_metabolism {
            return;
        }
        let world_size = self.config.world_size;

        let mut to_kill = Vec::new();
        for (org_idx, org) in self.organisms.iter_mut().enumerate() {
            if !org.alive {
                continue;
            }
            let center = if self.org_counts[org_idx] > 0 {
                [
                    Self::toroidal_mean_coord(
                        self.org_toroidal_sums[org_idx][0],
                        self.org_toroidal_sums[org_idx][1],
                        world_size,
                    ),
                    Self::toroidal_mean_coord(
                        self.org_toroidal_sums[org_idx][2],
                        self.org_toroidal_sums[org_idx][3],
                        world_size,
                    ),
                ]
            } else {
                [0.0, 0.0]
            };
            let external = self.resource_field.get(center[0], center[1]);
            let pre_energy = org.metabolic_state.energy;
            let engine = org.metabolism_engine.as_ref().unwrap_or(&self.metabolism);
            let flux = engine.step(&mut org.metabolic_state, external, self.config.dt as f32);
            let energy_delta = org.metabolic_state.energy - pre_energy;
            if energy_delta > 0.0 {
                let growth_factor = if self.config.enable_growth {
                    org.developmental_program.stage_factors(org.maturity).2
                } else {
                    self.config.growth_immature_metabolic_efficiency
                        + org.maturity * (1.0 - self.config.growth_immature_metabolic_efficiency)
                };
                org.metabolic_state.energy = pre_energy
                    + energy_delta * growth_factor * self.config.metabolism_efficiency_multiplier;
            }
            if flux.consumed_external > 0.0 {
                let _ = self
                    .resource_field
                    .take(center[0], center[1], flux.consumed_external);
            }

            if org.metabolic_state.energy <= self.config.death_energy_threshold
                || org.boundary_integrity <= boundary_terminal_threshold
            {
                to_kill.push(org_idx);
            }
        }
        for org_idx in to_kill {
            self.mark_dead(org_idx);
        }
    }
}

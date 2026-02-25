use super::super::World;
use crate::config::BoundaryMode;

impl World {
    /// Update boundary integrity using homeostasis aggregates from the state phase.
    pub(in crate::world) fn step_boundary_phase(&mut self, boundary_terminal_threshold: f32) {
        if !self.config.enable_boundary_maintenance {
            return;
        }

        let mut to_kill = Vec::new();
        {
            let config = &self.config;
            let dt = config.dt as f32;
            let homeostasis_sums = &self.homeostasis_sums_buffer;
            let homeostasis_counts = &self.homeostasis_counts_buffer;

            for (org_idx, org) in self.organisms.iter_mut().enumerate() {
                if !org.alive {
                    org.boundary_integrity = 0.0;
                    continue;
                }

                let energy_deficit =
                    (config.metabolic_viability_floor - org.metabolic_state.energy).max(0.0);
                let decay = config.boundary_decay_base_rate
                    + config.boundary_decay_energy_scale
                        * (energy_deficit
                            + org.metabolic_state.waste * config.boundary_waste_pressure_scale);
                let homeostasis_factor = if homeostasis_counts[org_idx] > 0 {
                    homeostasis_sums[org_idx] / homeostasis_counts[org_idx] as f32
                } else {
                    0.5
                };
                let dev_boundary = if config.enable_growth {
                    org.developmental_program.stage_factors(org.maturity).0
                } else {
                    1.0
                };
                let (decay_mode_scale, repair_mode_scale) = match config.boundary_mode {
                    BoundaryMode::ScalarRepair => (1.0, 1.0),
                    BoundaryMode::SpatialHullFeedback => {
                        let cohesion = if self.org_counts[org_idx] > 0 {
                            let count = self.org_counts[org_idx] as f64;
                            let x_mag = (self.org_toroidal_sums[org_idx][0].powi(2)
                                + self.org_toroidal_sums[org_idx][1].powi(2))
                            .sqrt()
                                / count;
                            let y_mag = (self.org_toroidal_sums[org_idx][2].powi(2)
                                + self.org_toroidal_sums[org_idx][3].powi(2))
                            .sqrt()
                                / count;
                            (0.5 * (x_mag + y_mag)).clamp(0.0, 1.0) as f32
                        } else {
                            0.0
                        };
                        let repair_scale = config.spatial_hull_repair_base
                            + config.spatial_hull_repair_cohesion_scale * cohesion;
                        let decay_scale = (config.spatial_hull_decay_base
                            - config.spatial_hull_decay_cohesion_scale * cohesion)
                            .max(config.spatial_hull_decay_min);
                        (decay_scale, repair_scale)
                    }
                };
                let repair = (org.metabolic_state.energy
                    - org.metabolic_state.waste
                        * config.boundary_waste_pressure_scale
                        * config.boundary_repair_waste_penalty_scale)
                    .max(0.0)
                    * config.boundary_repair_rate
                    * homeostasis_factor
                    * dev_boundary
                    * repair_mode_scale;
                org.boundary_integrity = (org.boundary_integrity - decay * decay_mode_scale * dt
                    + repair * dt)
                    .clamp(0.0, 1.0);
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

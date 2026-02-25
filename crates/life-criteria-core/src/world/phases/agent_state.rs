use super::super::World;
use crate::config::HomeostasisMode;
use std::f64::consts::PI;

impl World {
    /// Apply movement + homeostasis updates for each alive agent and gather
    /// aggregates consumed by boundary + metabolism phases.
    pub(in crate::world) fn step_agent_state_phase(&mut self) {
        let org_count = self.organisms.len();
        if self.homeostasis_sums_buffer.len() != org_count {
            self.homeostasis_sums_buffer.resize(org_count, 0.0);
            self.homeostasis_counts_buffer.resize(org_count, 0);
        }
        self.homeostasis_sums_buffer.fill(0.0);
        self.homeostasis_counts_buffer.fill(0);

        self.org_toroidal_sums.fill([0.0, 0.0, 0.0, 0.0]);
        self.org_counts.fill(0);

        let config = &self.config;
        let world_size = config.world_size;
        let tau_over_world = (2.0 * PI) / world_size;

        let agents = &mut self.agents;
        let deltas = &self.deltas_buffer;
        let organisms = &self.organisms;
        let homeostasis_sums = &mut self.homeostasis_sums_buffer;
        let homeostasis_counts = &mut self.homeostasis_counts_buffer;
        let org_toroidal_sums = &mut self.org_toroidal_sums;
        let org_counts = &mut self.org_counts;

        for (agent, delta) in agents.iter_mut().zip(deltas.iter()) {
            let org_idx = agent.organism_id as usize;
            if !organisms[org_idx].alive {
                agent.velocity = [0.0, 0.0];
                continue;
            }
            // Expose boundary with a one-step lag to avoid an extra full pass.
            agent.internal_state[2] = organisms[org_idx].boundary_integrity;

            if config.enable_response {
                agent.velocity[0] += delta[0] as f64 * config.dt;
                agent.velocity[1] += delta[1] as f64 * config.dt;
            }

            let speed_sq =
                agent.velocity[0] * agent.velocity[0] + agent.velocity[1] * agent.velocity[1];
            if speed_sq > config.max_speed * config.max_speed {
                let scale = config.max_speed / speed_sq.sqrt();
                agent.velocity[0] *= scale;
                agent.velocity[1] *= scale;
            }

            agent.position[0] =
                (agent.position[0] + agent.velocity[0] * config.dt).rem_euclid(config.world_size);
            agent.position[1] =
                (agent.position[1] + agent.velocity[1] * config.dt).rem_euclid(config.world_size);

            let h_decay = config.homeostasis_decay_rate * config.dt as f32;
            agent.internal_state[0] = (agent.internal_state[0] - h_decay).max(0.0);
            agent.internal_state[1] = (agent.internal_state[1] - h_decay).max(0.0);

            if config.enable_homeostasis {
                match config.homeostasis_mode {
                    HomeostasisMode::NnRegulator => {
                        agent.internal_state[0] =
                            (agent.internal_state[0] + delta[2] * config.dt as f32).clamp(0.0, 1.0);
                        agent.internal_state[1] =
                            (agent.internal_state[1] + delta[3] * config.dt as f32).clamp(0.0, 1.0);
                    }
                    HomeostasisMode::SetpointPid => {
                        let metabolic_energy =
                            organisms[org_idx].metabolic_state.energy.clamp(0.0, 1.0);
                        let setpoint = config.setpoint_pid_base
                            + config.setpoint_pid_energy_scale * metabolic_energy;
                        let adjustment_scale = config.setpoint_pid_kp * config.dt as f32;
                        let err0 = setpoint - agent.internal_state[0];
                        let err1 = setpoint - agent.internal_state[1];
                        agent.internal_state[0] =
                            (agent.internal_state[0] + err0 * adjustment_scale).clamp(0.0, 1.0);
                        agent.internal_state[1] =
                            (agent.internal_state[1] + err1 * adjustment_scale).clamp(0.0, 1.0);
                    }
                }
            }

            homeostasis_sums[org_idx] += agent.internal_state[0];
            homeostasis_counts[org_idx] += 1;

            let theta_x = agent.position[0] * tau_over_world;
            let theta_y = agent.position[1] * tau_over_world;
            let (sin_x, cos_x) = theta_x.sin_cos();
            let (sin_y, cos_y) = theta_y.sin_cos();
            org_toroidal_sums[org_idx][0] += sin_x;
            org_toroidal_sums[org_idx][1] += cos_x;
            org_toroidal_sums[org_idx][2] += sin_y;
            org_toroidal_sums[org_idx][3] += cos_y;
            org_counts[org_idx] += 1;
        }
    }
}

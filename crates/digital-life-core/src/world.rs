use crate::agent::Agent;
use crate::config::SimConfig;
use crate::metabolism::{MetabolicState, ToyMetabolism};
use crate::nn::NeuralNet;
use crate::spatial;
use std::time::Instant;
use std::{error::Error, fmt};

#[derive(Clone, Debug)]
pub struct StepTimings {
    pub spatial_build_us: u64,
    pub nn_query_us: u64,
    pub state_update_us: u64,
    pub total_us: u64,
}

pub struct World {
    pub agents: Vec<Agent>,
    pub nns: Vec<NeuralNet>, // one per organism
    pub config: SimConfig,
    metabolic_states: Vec<MetabolicState>,
    metabolism: ToyMetabolism,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorldInitError {
    InvalidWorldSize,
    InvalidDt,
    InvalidMaxSpeed,
    InvalidSensingRadius,
    InvalidNeighborNorm,
    NumOrganismsMismatch { expected: usize, actual: usize },
    InvalidOrganismId,
}

impl fmt::Display for WorldInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorldInitError::InvalidWorldSize => write!(f, "world_size must be positive and finite"),
            WorldInitError::InvalidDt => write!(f, "dt must be positive and finite"),
            WorldInitError::InvalidMaxSpeed => write!(f, "max_speed must be positive and finite"),
            WorldInitError::InvalidSensingRadius => {
                write!(f, "sensing_radius must be non-negative and finite")
            }
            WorldInitError::InvalidNeighborNorm => {
                write!(f, "neighbor_norm must be positive and finite")
            }
            WorldInitError::NumOrganismsMismatch { expected, actual } => write!(
                f,
                "num_organisms ({expected}) must match nns.len() ({actual})"
            ),
            WorldInitError::InvalidOrganismId => {
                write!(f, "all agent organism_ids must be valid indices into nns")
            }
        }
    }
}

impl Error for WorldInitError {}

impl World {
    pub fn new(agents: Vec<Agent>, nns: Vec<NeuralNet>, config: SimConfig) -> Self {
        Self::try_new(agents, nns, config).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_new(
        agents: Vec<Agent>,
        nns: Vec<NeuralNet>,
        config: SimConfig,
    ) -> Result<Self, WorldInitError> {
        if !(config.world_size.is_finite() && config.world_size > 0.0) {
            return Err(WorldInitError::InvalidWorldSize);
        }
        if !(config.dt.is_finite() && config.dt > 0.0) {
            return Err(WorldInitError::InvalidDt);
        }
        if !(config.max_speed.is_finite() && config.max_speed > 0.0) {
            return Err(WorldInitError::InvalidMaxSpeed);
        }
        if !(config.sensing_radius.is_finite() && config.sensing_radius >= 0.0) {
            return Err(WorldInitError::InvalidSensingRadius);
        }
        if !(config.neighbor_norm.is_finite() && config.neighbor_norm > 0.0) {
            return Err(WorldInitError::InvalidNeighborNorm);
        }
        if config.num_organisms != nns.len() {
            return Err(WorldInitError::NumOrganismsMismatch {
                expected: config.num_organisms,
                actual: nns.len(),
            });
        }
        if !agents.iter().all(|a| (a.organism_id as usize) < nns.len()) {
            return Err(WorldInitError::InvalidOrganismId);
        }

        let metabolic_states = vec![MetabolicState::default(); nns.len()];
        Ok(Self {
            agents,
            nns,
            config,
            metabolic_states,
            metabolism: ToyMetabolism::default(),
        })
    }

    pub fn metabolic_state(&self, organism_id: usize) -> &MetabolicState {
        self.try_metabolic_state(organism_id)
            .expect("organism_id out of range for metabolic_state")
    }

    pub fn try_metabolic_state(&self, organism_id: usize) -> Option<&MetabolicState> {
        self.metabolic_states.get(organism_id)
    }

    pub fn step(&mut self) -> StepTimings {
        let total_start = Instant::now();

        // 1. Build spatial index
        let t0 = Instant::now();
        let tree = spatial::build_index(&self.agents);
        let spatial_build_us = t0.elapsed().as_micros() as u64;

        // 2. NN forward pass for each agent
        let t1 = Instant::now();
        let mut deltas: Vec<[f32; 4]> = Vec::with_capacity(self.agents.len());
        for agent in &self.agents {
            let neighbor_count = spatial::count_neighbors(
                &tree,
                agent.position,
                self.config.sensing_radius,
                agent.id,
                self.config.world_size,
            );

            // Build NN input: position(2) + velocity(2) + internal_state(3) + neighbor_count(1)
            // internal_state[2] is a constant bias channel (read but not written by NN)
            // internal_state[3] is reserved for future criteria
            let input: [f32; 8] = [
                (agent.position[0] / self.config.world_size) as f32,
                (agent.position[1] / self.config.world_size) as f32,
                (agent.velocity[0] / self.config.max_speed) as f32,
                (agent.velocity[1] / self.config.max_speed) as f32,
                agent.internal_state[0],
                agent.internal_state[1],
                agent.internal_state[2],
                neighbor_count as f32 / self.config.neighbor_norm,
            ];

            let nn = &self.nns[agent.organism_id as usize];
            deltas.push(nn.forward(&input));
        }
        let nn_query_us = t1.elapsed().as_micros() as u64;

        // 3. Apply updates
        let t2 = Instant::now();
        for (agent, delta) in self.agents.iter_mut().zip(deltas.iter()) {
            // Velocity update
            agent.velocity[0] += delta[0] as f64 * self.config.dt;
            agent.velocity[1] += delta[1] as f64 * self.config.dt;

            // Clamp speed
            let speed_sq =
                agent.velocity[0] * agent.velocity[0] + agent.velocity[1] * agent.velocity[1];
            if speed_sq > self.config.max_speed * self.config.max_speed {
                let scale = self.config.max_speed / speed_sq.sqrt();
                agent.velocity[0] *= scale;
                agent.velocity[1] *= scale;
            }

            // Position update with toroidal wrapping
            agent.position[0] = (agent.position[0] + agent.velocity[0] * self.config.dt)
                .rem_euclid(self.config.world_size);
            agent.position[1] = (agent.position[1] + agent.velocity[1] * self.config.dt)
                .rem_euclid(self.config.world_size);

            // Internal state update (clamped to [0, 1])
            agent.internal_state[0] =
                (agent.internal_state[0] + delta[2] * self.config.dt as f32).clamp(0.0, 1.0);
            agent.internal_state[1] =
                (agent.internal_state[1] + delta[3] * self.config.dt as f32).clamp(0.0, 1.0);
        }

        // Update per-organism toy metabolism state.
        for state in &mut self.metabolic_states {
            self.metabolism.step(state, self.config.dt as f32);
        }
        let state_update_us = t2.elapsed().as_micros() as u64;

        StepTimings {
            spatial_build_us,
            nn_query_us,
            state_update_us,
            total_us: total_start.elapsed().as_micros() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimConfig;
    use crate::nn::NeuralNet;

    fn make_world(num_agents: usize, world_size: f64) -> World {
        let agents: Vec<Agent> = (0..num_agents)
            .map(|i| Agent::new(i as u32, 0, [50.0, 50.0]))
            .collect();
        let nn = NeuralNet::from_weights(std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT));
        let config = SimConfig {
            world_size,
            num_organisms: 1,
            ..SimConfig::default()
        };
        World::new(agents, vec![nn], config)
    }

    fn make_config(world_size: f64, dt: f64) -> SimConfig {
        SimConfig {
            world_size,
            dt,
            num_organisms: 1,
            ..SimConfig::default()
        }
    }

    #[test]
    fn toroidal_wrapping_keeps_positions_in_bounds() {
        let mut world = make_world(1, 100.0);
        world.agents[0].velocity = [100.0, 100.0]; // will overshoot in one step
        world.step();
        let pos = world.agents[0].position;
        assert!(
            pos[0] >= 0.0 && pos[0] < 100.0,
            "x={} out of bounds",
            pos[0]
        );
        assert!(
            pos[1] >= 0.0 && pos[1] < 100.0,
            "y={} out of bounds",
            pos[1]
        );
    }

    #[test]
    fn step_returns_nonzero_timings() {
        let mut world = make_world(10, 100.0);
        let t = world.step();
        assert!(t.total_us > 0);
    }

    #[test]
    #[should_panic(expected = "organism_ids must be valid")]
    fn new_panics_on_invalid_organism_id() {
        let agents = vec![Agent::new(0, 5, [0.0, 0.0])]; // organism_id=5, but only 1 NN
        let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
        World::new(agents, vec![nn], make_config(100.0, 0.1));
    }

    #[test]
    #[should_panic(expected = "world_size must be positive and finite")]
    fn new_panics_on_non_positive_world_size() {
        let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
        let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
        World::new(agents, vec![nn], make_config(0.0, 0.1));
    }

    #[test]
    #[should_panic(expected = "world_size must be positive and finite")]
    fn new_panics_on_non_finite_world_size() {
        let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
        let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
        World::new(agents, vec![nn], make_config(f64::NAN, 0.1));
    }

    #[test]
    #[should_panic(expected = "num_organisms")]
    fn new_panics_on_num_organisms_mismatch() {
        let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
        let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
        let mut cfg = make_config(100.0, 0.1);
        cfg.num_organisms = 2;
        World::new(agents, vec![nn], cfg);
    }

    #[test]
    fn try_new_returns_structured_error() {
        let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
        let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
        let mut cfg = make_config(100.0, 0.1);
        cfg.num_organisms = 2;
        match World::try_new(agents, vec![nn], cfg) {
            Err(WorldInitError::NumOrganismsMismatch {
                expected: 2,
                actual: 1,
            }) => {}
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("expected try_new to fail for mismatched organism count"),
        }
    }

    #[test]
    fn internal_state_stays_clamped() {
        let mut world = make_world(1, 100.0);
        // Run many steps — state should remain in [0, 1]
        for _ in 0..100 {
            world.step();
        }
        for &s in &world.agents[0].internal_state {
            assert!(
                (0.0..=1.0).contains(&s),
                "internal_state {s} out of [0,1] range"
            );
        }
    }

    #[test]
    fn step_respects_config_dt_for_position_update() {
        let mut world = make_world(1, 100.0);
        world.agents[0].position = [50.0, 50.0];
        world.agents[0].velocity = [1.0, 0.0];
        world.nns[0] =
            NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
        world.config.dt = 0.5;
        world.step();
        assert!(
            (world.agents[0].position[0] - 50.5).abs() < 1e-6,
            "expected x to advance by dt-scaled velocity"
        );
    }

    #[test]
    fn toy_metabolism_sustains_energy_for_1000_steps() {
        let mut world = make_world(10, 100.0);
        for _ in 0..1000 {
            world.step();
        }
        assert!(world.metabolic_state(0).energy > 0.0);
    }

    #[test]
    fn try_metabolic_state_returns_none_for_out_of_range() {
        let world = make_world(1, 100.0);
        assert!(world.try_metabolic_state(10).is_none());
    }
}

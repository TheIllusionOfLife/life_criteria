use crate::agent::Agent;
use crate::nn::NeuralNet;
use crate::spatial;
use std::time::Instant;

const SENSING_RADIUS: f64 = 5.0;
const MAX_SPEED: f64 = 2.0;
const DT: f64 = 0.1;

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
    pub world_size: f64,
    pub num_organisms: usize,
}

impl World {
    pub fn new(
        agents: Vec<Agent>,
        nns: Vec<NeuralNet>,
        world_size: f64,
        num_organisms: usize,
    ) -> Self {
        assert!(
            agents
                .iter()
                .all(|a| (a.organism_id as usize) < nns.len()),
            "all agent organism_ids must be valid indices into nns"
        );
        Self {
            agents,
            nns,
            world_size,
            num_organisms,
        }
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
            let _neighbors = spatial::query_neighbors(&tree, agent.position, SENSING_RADIUS);
            let neighbor_count = _neighbors.len() as f32;

            // Build NN input: position(2) + velocity(2) + internal_state(3) + neighbor_count(1)
            let input: [f32; 8] = [
                (agent.position[0] / self.world_size) as f32,
                (agent.position[1] / self.world_size) as f32,
                (agent.velocity[0] / MAX_SPEED) as f32,
                (agent.velocity[1] / MAX_SPEED) as f32,
                agent.internal_state[0],
                agent.internal_state[1],
                agent.internal_state[2],
                neighbor_count / 50.0,
            ];

            let nn = &self.nns[agent.organism_id as usize];
            deltas.push(nn.forward(&input));
        }
        let nn_query_us = t1.elapsed().as_micros() as u64;

        // 3. Apply updates
        let t2 = Instant::now();
        for (agent, delta) in self.agents.iter_mut().zip(deltas.iter()) {
            // Velocity update
            agent.velocity[0] += delta[0] as f64 * DT;
            agent.velocity[1] += delta[1] as f64 * DT;

            // Clamp speed
            let speed_sq = agent.velocity[0] * agent.velocity[0]
                + agent.velocity[1] * agent.velocity[1];
            if speed_sq > MAX_SPEED * MAX_SPEED {
                let scale = MAX_SPEED / speed_sq.sqrt();
                agent.velocity[0] *= scale;
                agent.velocity[1] *= scale;
            }

            // Position update with toroidal wrapping
            agent.position[0] =
                (agent.position[0] + agent.velocity[0] * DT).rem_euclid(self.world_size);
            agent.position[1] =
                (agent.position[1] + agent.velocity[1] * DT).rem_euclid(self.world_size);

            // Internal state update (clamped to [0, 1])
            agent.internal_state[0] = (agent.internal_state[0] + delta[2] * DT as f32).clamp(0.0, 1.0);
            agent.internal_state[1] = (agent.internal_state[1] + delta[3] * DT as f32).clamp(0.0, 1.0);
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
    use crate::nn::NeuralNet;

    fn make_world(num_agents: usize, world_size: f64) -> World {
        let agents: Vec<Agent> = (0..num_agents)
            .map(|i| Agent::new(i as u32, 0, [50.0, 50.0]))
            .collect();
        let nn = NeuralNet::from_weights(
            std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT),
        );
        World::new(agents, vec![nn], world_size, 1)
    }

    #[test]
    fn toroidal_wrapping_keeps_positions_in_bounds() {
        let mut world = make_world(1, 100.0);
        world.agents[0].velocity = [100.0, 100.0]; // will overshoot in one step
        world.step();
        let pos = world.agents[0].position;
        assert!(
            pos[0] >= 0.0 && pos[0] < 100.0,
            "x={} out of bounds", pos[0]
        );
        assert!(
            pos[1] >= 0.0 && pos[1] < 100.0,
            "y={} out of bounds", pos[1]
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
        let nn = NeuralNet::from_weights(
            std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT),
        );
        World::new(agents, vec![nn], 100.0, 1);
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
}

use crate::genome::Genome;
use crate::nn::NeuralNet;

#[derive(Clone, Debug)]
pub struct Organism {
    // Fields are private by design; use accessors to preserve invariants.
    id: u16,
    agent_start: usize,
    agent_count: usize,
    nn: NeuralNet,
    genome: Genome,
}

impl Organism {
    pub fn new(
        id: u16,
        agent_start: usize,
        agent_count: usize,
        nn: NeuralNet,
        genome: Genome,
    ) -> Self {
        Self {
            id,
            agent_start,
            agent_count,
            nn,
            genome,
        }
    }

    pub fn agent_range(&self) -> std::ops::Range<usize> {
        self.agent_start..self.agent_start + self.agent_count
    }

    pub fn id(&self) -> u16 {
        self.id
    }

    pub fn agent_start(&self) -> usize {
        self.agent_start
    }

    pub fn agent_count(&self) -> usize {
        self.agent_count
    }

    pub fn nn(&self) -> &NeuralNet {
        &self.nn
    }

    pub fn genome(&self) -> &Genome {
        &self.genome
    }
}

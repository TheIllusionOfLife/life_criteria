/// Per-organism metabolic state.
#[derive(Clone, Debug)]
pub struct MetabolicState {
    pub energy: f32,
    pub resource: f32,
    pub waste: f32,
}

impl Default for MetabolicState {
    fn default() -> Self {
        Self {
            energy: 0.5,
            resource: 5.0,
            waste: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetabolismFlux {
    pub consumed_external: f32,
    pub consumed_total: f32,
    pub produced_waste: f32,
}

/// Future-facing node definition for genetically encoded graph metabolism.
#[derive(Clone, Debug)]
pub struct MetabolicNode {
    pub id: u16,
    pub catalytic_efficiency: f32,
}

/// Future-facing edge definition for graph metabolism.
#[derive(Clone, Debug)]
pub struct MetabolicEdge {
    pub from: u16,
    pub to: u16,
    pub flux_ratio: f32,
}

/// Graph topology scaffold used by the graph metabolism strategy.
#[derive(Clone, Debug)]
pub struct MetabolicGraph {
    pub nodes: Vec<MetabolicNode>,
    pub edges: Vec<MetabolicEdge>,
}

impl MetabolicGraph {
    pub fn bootstrap_single_path() -> Self {
        Self {
            nodes: vec![MetabolicNode {
                id: 0,
                catalytic_efficiency: 1.0,
            }],
            edges: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ToyMetabolism {
    pub uptake_rate: f32,
    pub conversion_efficiency: f32,
    pub waste_ratio: f32,
    pub energy_loss_rate: f32,
    pub max_energy: f32,
    pub waste_decay_rate: f32,
    pub max_waste: f32,
}

impl Default for ToyMetabolism {
    fn default() -> Self {
        Self {
            uptake_rate: 0.4,
            conversion_efficiency: 0.8,
            waste_ratio: 0.2,
            energy_loss_rate: 0.02,
            max_energy: 1.0,
            waste_decay_rate: 0.05,
            max_waste: 1.0,
        }
    }
}

impl ToyMetabolism {
    pub fn step(
        &self,
        state: &mut MetabolicState,
        external_resource: f32,
        dt: f32,
    ) -> MetabolismFlux {
        let params = MetabolismParams::from_toy(self);
        apply_metabolism_step(
            params,
            self.conversion_efficiency,
            state,
            external_resource,
            dt,
        )
    }
}

#[derive(Clone, Debug)]
pub struct GraphMetabolism {
    pub graph: MetabolicGraph,
    pub uptake_rate: f32,
    pub conversion_efficiency: f32,
    pub waste_ratio: f32,
    pub energy_loss_rate: f32,
    pub max_energy: f32,
    pub waste_decay_rate: f32,
    pub max_waste: f32,
}

impl Default for GraphMetabolism {
    fn default() -> Self {
        let toy = ToyMetabolism::default();
        Self {
            graph: MetabolicGraph::bootstrap_single_path(),
            uptake_rate: toy.uptake_rate,
            conversion_efficiency: toy.conversion_efficiency,
            waste_ratio: toy.waste_ratio,
            energy_loss_rate: toy.energy_loss_rate,
            max_energy: toy.max_energy,
            waste_decay_rate: toy.waste_decay_rate,
            max_waste: toy.max_waste,
        }
    }
}

impl GraphMetabolism {
    pub fn step(
        &self,
        state: &mut MetabolicState,
        external_resource: f32,
        dt: f32,
    ) -> MetabolismFlux {
        // Bootstrap behavior: graph topology modulates conversion efficiency.
        let node_efficiency = if self.graph.nodes.is_empty() {
            1.0
        } else {
            self.graph
                .nodes
                .iter()
                .map(|node| node.catalytic_efficiency)
                .sum::<f32>()
                / self.graph.nodes.len() as f32
        };
        let converted_eff = (self.conversion_efficiency * node_efficiency).clamp(0.0, 1.0);
        let params = MetabolismParams::from_graph(self);
        apply_metabolism_step(params, converted_eff, state, external_resource, dt)
    }
}

#[derive(Clone, Debug)]
pub enum MetabolismEngine {
    Toy(ToyMetabolism),
    Graph(GraphMetabolism),
}

impl Default for MetabolismEngine {
    fn default() -> Self {
        Self::Toy(ToyMetabolism::default())
    }
}

impl MetabolismEngine {
    pub fn step(
        &self,
        state: &mut MetabolicState,
        external_resource: f32,
        dt: f32,
    ) -> MetabolismFlux {
        match self {
            MetabolismEngine::Toy(engine) => engine.step(state, external_resource, dt),
            MetabolismEngine::Graph(engine) => engine.step(state, external_resource, dt),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MetabolismParams {
    uptake_rate: f32,
    waste_ratio: f32,
    energy_loss_rate: f32,
    max_energy: f32,
    waste_decay_rate: f32,
    max_waste: f32,
}

impl MetabolismParams {
    fn from_toy(engine: &ToyMetabolism) -> Self {
        Self {
            uptake_rate: engine.uptake_rate,
            waste_ratio: engine.waste_ratio,
            energy_loss_rate: engine.energy_loss_rate,
            max_energy: engine.max_energy,
            waste_decay_rate: engine.waste_decay_rate,
            max_waste: engine.max_waste,
        }
    }

    fn from_graph(engine: &GraphMetabolism) -> Self {
        Self {
            uptake_rate: engine.uptake_rate,
            waste_ratio: engine.waste_ratio,
            energy_loss_rate: engine.energy_loss_rate,
            max_energy: engine.max_energy,
            waste_decay_rate: engine.waste_decay_rate,
            max_waste: engine.max_waste,
        }
    }
}

fn apply_metabolism_step(
    params: MetabolismParams,
    conversion_efficiency: f32,
    state: &mut MetabolicState,
    external_resource: f32,
    dt: f32,
) -> MetabolismFlux {
    let external_cap = (params.uptake_rate * dt).max(0.0);
    let consumed_external = external_resource.max(0.0).min(external_cap);
    state.resource += consumed_external;

    let uptake = (params.uptake_rate * dt).min(state.resource).max(0.0);
    state.resource -= uptake;
    state.energy += uptake * conversion_efficiency;
    state.waste += uptake * params.waste_ratio;
    state.waste = (state.waste - params.waste_decay_rate * dt).clamp(0.0, params.max_waste);

    // Minimal thermodynamic loss to avoid unbounded free energy growth.
    let retained = (1.0 - params.energy_loss_rate * dt).clamp(0.0, 1.0);
    state.energy = (state.energy * retained).clamp(0.0, params.max_energy);

    MetabolismFlux {
        consumed_external,
        consumed_total: uptake,
        produced_waste: uptake * params.waste_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_is_bounded() {
        let mut state = MetabolicState::default();
        let metabolism = ToyMetabolism {
            uptake_rate: 10.0,
            conversion_efficiency: 1.0,
            energy_loss_rate: 0.0,
            ..ToyMetabolism::default()
        };
        for _ in 0..100 {
            state.resource += 10.0;
            metabolism.step(&mut state, 0.0, 1.0);
        }
        assert!(
            (0.0..=metabolism.max_energy).contains(&state.energy),
            "energy out of bounds: {}",
            state.energy
        );
    }

    #[test]
    fn waste_is_bounded() {
        let mut state = MetabolicState::default();
        let metabolism = ToyMetabolism {
            uptake_rate: 10.0,
            waste_ratio: 1.0,
            waste_decay_rate: 0.0,
            ..ToyMetabolism::default()
        };
        for _ in 0..100 {
            state.resource += 10.0;
            metabolism.step(&mut state, 0.0, 1.0);
        }
        assert!(
            (0.0..=metabolism.max_waste).contains(&state.waste),
            "waste out of bounds: {}",
            state.waste
        );
    }

    #[test]
    fn external_resource_is_consumed() {
        let mut state = MetabolicState {
            resource: 0.0,
            ..MetabolicState::default()
        };
        let metabolism = ToyMetabolism::default();
        let flux = metabolism.step(&mut state, 1.0, 1.0);
        assert!(flux.consumed_external > 0.0);
        assert!(state.energy > 0.0);
    }

    #[test]
    fn graph_engine_produces_bounded_state() {
        let mut state = MetabolicState::default();
        let metabolism = GraphMetabolism::default();
        for _ in 0..100 {
            let _ = metabolism.step(&mut state, 1.0, 1.0);
        }
        assert!((0.0..=metabolism.max_energy).contains(&state.energy));
        assert!((0.0..=metabolism.max_waste).contains(&state.waste));
    }
}

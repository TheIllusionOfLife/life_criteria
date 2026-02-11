use crate::agent::Agent;
use rstar::{RTree, AABB};

/// Build an R*-tree from agents via bulk_load (O(n log n)).
pub fn build_index(agents: &[Agent]) -> RTree<Agent> {
    RTree::bulk_load(agents.to_vec())
}

/// Query neighbors within `radius` of `center`, returning their agent IDs.
/// Uses AABB envelope query then filters by Euclidean distance.
pub fn query_neighbors(
    tree: &RTree<Agent>,
    center: [f64; 2],
    radius: f64,
) -> Vec<u32> {
    let envelope = AABB::from_corners(
        [center[0] - radius, center[1] - radius],
        [center[0] + radius, center[1] + radius],
    );
    let r_sq = radius * radius;

    tree.locate_in_envelope(&envelope)
        .filter(|agent| {
            let dx = agent.position[0] - center[0];
            let dy = agent.position[1] - center[1];
            dx * dx + dy * dy <= r_sq
        })
        .map(|agent| agent.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent(id: u32, x: f64, y: f64) -> Agent {
        Agent::new(id, 0, [x, y])
    }

    #[test]
    fn query_finds_agents_within_radius() {
        let agents = vec![
            make_agent(0, 5.0, 5.0),
            make_agent(1, 6.0, 5.0),  // distance 1.0
            make_agent(2, 50.0, 50.0), // far away
        ];
        let tree = build_index(&agents);
        let mut result = query_neighbors(&tree, [5.0, 5.0], 2.0);
        result.sort();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn query_excludes_agents_outside_radius() {
        let agents = vec![
            make_agent(0, 0.0, 0.0),
            make_agent(1, 10.0, 10.0),
        ];
        let tree = build_index(&agents);
        let result = query_neighbors(&tree, [0.0, 0.0], 1.0);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn query_returns_agent_ids_not_indices() {
        let agents = vec![
            make_agent(42, 1.0, 1.0),
            make_agent(99, 1.5, 1.0),
        ];
        let tree = build_index(&agents);
        let mut result = query_neighbors(&tree, [1.0, 1.0], 2.0);
        result.sort();
        assert_eq!(result, vec![42, 99]);
    }
}

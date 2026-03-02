use crate::agent::Agent;

/// Lightweight entry stored in the spatial grid.
#[derive(Clone, Debug)]
pub struct GridEntry {
    pub id: u32,
    pub position: [f64; 2],
}

/// Uniform grid spatial index with CSR-style flat layout.
///
/// Divides the toroidal world into `grid_dim × grid_dim` cells of size ≥ `sensing_radius`.
/// Neighbor queries scan a 3×3 cell neighborhood — O(k) where k = agents in those cells.
/// Build is O(n) via counting sort (two passes over agents).
///
/// The flat CSR layout stores all entries in a single contiguous `Vec<GridEntry>`,
/// with `offsets[cell] .. offsets[cell+1]` indexing into it. This avoids N+1 heap
/// allocations from `Vec<Vec<>>` and is much more cache-friendly.
pub struct UniformGrid {
    /// CSR offsets: `offsets[cell] .. offsets[cell+1]` indexes into `entries`.
    /// Length = `grid_dim * grid_dim + 1`.
    offsets: Vec<usize>,
    /// Contiguous agent entries, grouped by cell.
    entries: Vec<GridEntry>,
    cell_size: f64,
    grid_dim: usize,
}

impl UniformGrid {
    /// Build a grid from an iterator of (id, position) pairs.
    fn build(
        agents: impl Iterator<Item = (u32, [f64; 2])> + Clone,
        sensing_radius: f64,
        world_size: f64,
    ) -> Self {
        let cell_size = sensing_radius.max(1.0);
        let grid_dim = (world_size / cell_size).floor() as usize;
        let grid_dim = grid_dim.max(1);
        let num_cells = grid_dim * grid_dim;

        // Pass 1: count agents per cell
        let mut counts = vec![0usize; num_cells];
        let mut total = 0usize;
        for (_id, pos) in agents.clone() {
            let cell = cell_index(pos, cell_size, grid_dim);
            counts[cell] += 1;
            total += 1;
        }

        // Prefix sum → offsets
        let mut offsets = vec![0usize; num_cells + 1];
        for i in 0..num_cells {
            offsets[i + 1] = offsets[i] + counts[i];
        }

        // Pass 2: place entries
        let mut entries = vec![
            GridEntry {
                id: 0,
                position: [0.0; 2],
            };
            total
        ];
        let mut cursors = vec![0usize; num_cells];
        for (id, position) in agents {
            let cell = cell_index(position, cell_size, grid_dim);
            let idx = offsets[cell] + cursors[cell];
            entries[idx] = GridEntry { id, position };
            cursors[cell] += 1;
        }

        Self {
            offsets,
            entries,
            cell_size,
            grid_dim,
        }
    }

    /// Iterate entries in a given cell.
    #[inline]
    fn cell_entries(&self, cell: usize) -> &[GridEntry] {
        let start = self.offsets[cell];
        let end = self.offsets[cell + 1];
        &self.entries[start..end]
    }
}

#[inline]
fn cell_index(pos: [f64; 2], cell_size: f64, grid_dim: usize) -> usize {
    let cx = ((pos[0] / cell_size) as usize).min(grid_dim - 1);
    let cy = ((pos[1] / cell_size) as usize).min(grid_dim - 1);
    cy * grid_dim + cx
}

#[inline]
fn wrapped_delta(delta: f64, world_size: f64) -> f64 {
    (delta + world_size / 2.0).rem_euclid(world_size) - world_size / 2.0
}

/// Iterate over unique neighbors, calling `visitor` for each.
fn for_each_unique_neighbor(
    grid: &UniformGrid,
    center: [f64; 2],
    radius: f64,
    self_id: u32,
    world_size: f64,
    mut visitor: impl FnMut(u32),
) {
    assert!(
        world_size.is_finite() && world_size > 0.0,
        "world_size must be positive and finite"
    );
    let r_sq = radius * radius;
    let cell_size = grid.cell_size;
    let grid_dim = grid.grid_dim;
    let cx = (center[0] / cell_size) as isize;
    let cy = (center[1] / cell_size) as isize;
    let gd = grid_dim as isize;

    // Scan enough cells to cover the query radius.
    // When radius ≤ cell_size this is 1 → 3×3 (the common case).
    // For larger effective radii (e.g. growth-boosted sensing), we widen the scan.
    let extent = (radius / cell_size).ceil() as isize;

    // When 2*extent+1 > grid_dim, multiple (dx,dy) offsets wrap to the same cell.
    // Track visited cells to avoid counting agents multiple times.
    let scan_width = (2 * extent + 1) as usize;
    let needs_dedup = scan_width > grid_dim;
    let num_cells = grid_dim * grid_dim;
    let mut visited = if needs_dedup {
        vec![false; num_cells]
    } else {
        Vec::new()
    };

    for dy in -extent..=extent {
        for dx in -extent..=extent {
            let nx = (cx + dx).rem_euclid(gd);
            let ny = (cy + dy).rem_euclid(gd);
            let cell = (ny as usize) * grid_dim + (nx as usize);

            if needs_dedup {
                if visited[cell] {
                    continue;
                }
                visited[cell] = true;
            }

            for entry in grid.cell_entries(cell) {
                if entry.id == self_id {
                    continue;
                }
                let ex = wrapped_delta(entry.position[0] - center[0], world_size);
                let ey = wrapped_delta(entry.position[1] - center[1], world_size);
                if ex * ex + ey * ey <= r_sq {
                    visitor(entry.id);
                }
            }
        }
    }
}

/// Build a uniform grid from all agent positions.
pub fn build_index(agents: &[Agent], sensing_radius: f64, world_size: f64) -> UniformGrid {
    UniformGrid::build(
        agents.iter().map(|a| (a.id, a.position)),
        sensing_radius,
        world_size,
    )
}

/// Build a uniform grid from only agents belonging to alive organisms.
pub fn build_index_active(
    agents: &[Agent],
    organism_alive: &[bool],
    sensing_radius: f64,
    world_size: f64,
) -> UniformGrid {
    UniformGrid::build(
        agents
            .iter()
            .filter(|a| {
                organism_alive
                    .get(a.organism_id as usize)
                    .copied()
                    .unwrap_or(false)
            })
            .map(|a| (a.id, a.position)),
        sensing_radius,
        world_size,
    )
}

/// Count neighbors within `radius` of `center` (excludes agent with `self_id`).
pub fn count_neighbors(
    grid: &UniformGrid,
    center: [f64; 2],
    radius: f64,
    self_id: u32,
    world_size: f64,
) -> usize {
    let mut count = 0usize;
    for_each_unique_neighbor(grid, center, radius, self_id, world_size, |_| {
        count += 1;
    });
    count
}

/// Query neighbors within `radius` of `center`, returning sorted agent IDs.
pub fn query_neighbors(
    grid: &UniformGrid,
    center: [f64; 2],
    radius: f64,
    self_id: u32,
    world_size: f64,
) -> Vec<u32> {
    let mut result = Vec::new();
    for_each_unique_neighbor(grid, center, radius, self_id, world_size, |id| {
        result.push(id);
    });
    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent(id: u32, x: f64, y: f64) -> Agent {
        Agent::new(id, 0, [x, y])
    }

    /// Helper: build grid with default radius=5.0, world=100.0
    fn build_default(agents: &[Agent]) -> UniformGrid {
        build_index(agents, 5.0, 100.0)
    }

    #[test]
    fn query_finds_agents_within_radius() {
        let agents = vec![
            make_agent(0, 5.0, 5.0),
            make_agent(1, 6.0, 5.0),   // distance 1.0
            make_agent(2, 50.0, 50.0), // far away
        ];
        let grid = build_default(&agents);
        let result = query_neighbors(&grid, [5.0, 5.0], 2.0, u32::MAX, 100.0);
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn query_excludes_self() {
        let agents = vec![make_agent(0, 5.0, 5.0), make_agent(1, 6.0, 5.0)];
        let grid = build_default(&agents);
        let result = query_neighbors(&grid, [5.0, 5.0], 2.0, 0, 100.0);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn query_excludes_agents_outside_radius() {
        let agents = vec![make_agent(0, 0.0, 0.0), make_agent(1, 10.0, 10.0)];
        let grid = build_default(&agents);
        let result = query_neighbors(&grid, [0.0, 0.0], 1.0, u32::MAX, 100.0);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn query_returns_agent_ids_not_indices() {
        let agents = vec![make_agent(42, 1.0, 1.0), make_agent(99, 1.5, 1.0)];
        let grid = build_default(&agents);
        let result = query_neighbors(&grid, [1.0, 1.0], 2.0, u32::MAX, 100.0);
        assert_eq!(result, vec![42, 99]);
    }

    #[test]
    fn count_neighbors_excludes_self() {
        let agents = vec![
            make_agent(0, 5.0, 5.0),
            make_agent(1, 6.0, 5.0),
            make_agent(2, 50.0, 50.0),
        ];
        let grid = build_default(&agents);
        assert_eq!(count_neighbors(&grid, [5.0, 5.0], 2.0, 0, 100.0), 1);
    }

    #[test]
    fn count_neighbors_wraps_toroidally_across_world_edges() {
        // world size 100: x=99.8 and x=0.5 are 0.7 apart toroidally
        let agents = vec![make_agent(0, 0.5, 50.0), make_agent(1, 99.8, 50.0)];
        let grid = build_default(&agents);
        assert_eq!(count_neighbors(&grid, [0.5, 50.0], 1.0, 0, 100.0), 1);
    }

    #[test]
    fn query_neighbors_wraps_toroidally_at_corner() {
        let agents = vec![make_agent(0, 0.2, 0.2), make_agent(1, 99.8, 99.8)];
        let grid = build_default(&agents);
        let result = query_neighbors(&grid, [0.2, 0.2], 1.0, 0, 100.0);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn query_neighbors_returns_sorted_unique_ids() {
        let agents = vec![
            make_agent(10, 0.2, 50.0),
            make_agent(2, 99.9, 50.0),
            make_agent(7, 0.4, 50.0),
        ];
        let grid = build_default(&agents);
        let result = query_neighbors(&grid, [0.1, 50.0], 1.0, u32::MAX, 100.0);
        assert_eq!(result, vec![2, 7, 10]);
    }

    #[test]
    fn build_index_active_excludes_inactive_organisms() {
        let agents = vec![Agent::new(0, 0, [1.0, 1.0]), Agent::new(1, 1, [1.0, 1.2])];
        let grid = build_index_active(&agents, &[true, false], 5.0, 100.0);
        let result = query_neighbors(&grid, [1.0, 1.0], 1.0, u32::MAX, 100.0);
        assert_eq!(result, vec![0]);
    }

    /// Brute-force neighbor computation for validation.
    fn brute_force_neighbors(
        agents: &[Agent],
        center: [f64; 2],
        radius: f64,
        self_id: u32,
        world_size: f64,
    ) -> Vec<u32> {
        let r_sq = radius * radius;
        let mut result: Vec<u32> = agents
            .iter()
            .filter(|a| {
                if a.id == self_id {
                    return false;
                }
                let dx = wrapped_delta(a.position[0] - center[0], world_size);
                let dy = wrapped_delta(a.position[1] - center[1], world_size);
                dx * dx + dy * dy <= r_sq
            })
            .map(|a| a.id)
            .collect();
        result.sort_unstable();
        result
    }

    #[test]
    fn grid_matches_brute_force() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let world_size = 100.0;
        let sensing_radius = 5.0;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Random agents
        let agents: Vec<Agent> = (0..500)
            .map(|i| {
                make_agent(
                    i,
                    rng.random::<f64>() * world_size,
                    rng.random::<f64>() * world_size,
                )
            })
            .collect();

        let grid = build_index(&agents, sensing_radius, world_size);

        // Test various radii including effective radii different from cell_size
        for &radius in &[1.0, 3.0, 5.0, 7.5, 10.0] {
            for agent in agents.iter().take(50) {
                let grid_result =
                    query_neighbors(&grid, agent.position, radius, agent.id, world_size);
                let brute_result =
                    brute_force_neighbors(&agents, agent.position, radius, agent.id, world_size);
                assert_eq!(
                    grid_result, brute_result,
                    "Mismatch for agent {} at {:?} radius={}",
                    agent.id, agent.position, radius
                );
            }
        }
    }

    #[test]
    fn small_grid_no_duplicate_neighbors() {
        // With world_size=5 and sensing_radius=5, grid_dim=1 (single cell).
        // All agents share one cell, so the scan wraps multiple offsets to cell 0.
        // Without dedup, agents would be visited multiple times.
        let agents = vec![
            make_agent(0, 1.0, 1.0),
            make_agent(1, 2.0, 2.0),
            make_agent(2, 3.0, 3.0),
        ];
        let grid = build_index(&agents, 5.0, 5.0);
        assert_eq!(grid.grid_dim, 1);
        let result = query_neighbors(&grid, [1.0, 1.0], 5.0, 0, 5.0);
        // Each agent should appear exactly once
        assert_eq!(result, vec![1, 2]);
        assert_eq!(count_neighbors(&grid, [1.0, 1.0], 5.0, 0, 5.0), 2);
    }

    #[test]
    fn small_grid_dim2_no_duplicate_neighbors() {
        // world_size=10, sensing_radius=5 → grid_dim=2, but extent=1 means 3×3 scan
        // wraps around the 2×2 grid, causing cell revisits without dedup.
        let agents = vec![make_agent(0, 1.0, 1.0), make_agent(1, 6.0, 6.0)];
        let grid = build_index(&agents, 5.0, 10.0);
        assert_eq!(grid.grid_dim, 2);
        // Distance between (1,1) and (6,6) toroidally = sqrt((5)^2+(5)^2) = 7.07
        // With radius=10 (covers whole world), both should appear once
        let result = query_neighbors(&grid, [1.0, 1.0], 10.0, 0, 10.0);
        assert_eq!(result, vec![1]);
        assert_eq!(count_neighbors(&grid, [1.0, 1.0], 10.0, 0, 10.0), 1);
    }

    #[test]
    fn bench_count_neighbors_near_boundary() {
        use std::time::Instant;
        let n_agents = 100_000;
        let world_size = 100.0;
        let radius = 5.0;

        let mut agents = Vec::with_capacity(n_agents);
        for i in 0..n_agents {
            let x = if i % 2 == 0 {
                (i as f64 / n_agents as f64) * 5.0
            } else {
                95.0 + (i as f64 / n_agents as f64) * 5.0
            };
            let y = (i as f64 / n_agents as f64) * world_size;
            agents.push(make_agent(i as u32, x, y));
        }

        let grid = build_index(&agents, radius, world_size);

        let start = Instant::now();
        let mut total_neighbors = 0;
        for _ in 0..10 {
            for agent in agents.iter().take(1000) {
                let center = agent.position;
                total_neighbors += count_neighbors(&grid, center, radius, agent.id, world_size);
            }
        }
        let duration = start.elapsed();
        println!(
            "Benchmark: counted {} neighbors in {:?}",
            total_neighbors, duration
        );
    }
}

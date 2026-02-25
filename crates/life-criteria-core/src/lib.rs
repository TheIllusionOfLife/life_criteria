pub mod agent;
pub mod config;
pub mod constants;
pub mod genome;
pub mod metabolism;
pub mod metrics;
pub mod nn;
pub mod organism;
pub mod resource;
pub mod rng;
pub mod spatial;
pub mod world;

pub use constants::MAX_WORLD_SIZE;
pub use metrics::{
    LineageEvent, OrganismSnapshot, PopulationStats, RunSummary, SnapshotFrame, StepMetrics,
};

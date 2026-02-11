/// 2D grid resource field stub.
/// Each cell holds a resource concentration value.

#[derive(Clone, Debug)]
pub struct ResourceField {
    width: usize,
    height: usize,
    cell_size: f64,
    data: Vec<f32>,
}

impl ResourceField {
    pub fn new(world_size: f64, cell_size: f64, initial_value: f32) -> Self {
        assert!(world_size > 0.0, "world_size must be positive");
        assert!(cell_size > 0.0, "cell_size must be positive");
        let width = (world_size / cell_size).ceil() as usize;
        let height = width;
        let data = vec![initial_value; width * height];
        Self {
            width,
            height,
            cell_size,
            data,
        }
    }

    /// Get resource value at position. Coordinates are clamped to grid bounds.
    pub fn get(&self, x: f64, y: f64) -> f32 {
        let (cx, cy) = self.clamp_coords(x, y);
        self.data[cy * self.width + cx]
    }

    /// Set resource value at position. Coordinates are clamped to grid bounds.
    pub fn set(&mut self, x: f64, y: f64, value: f32) {
        let (cx, cy) = self.clamp_coords(x, y);
        self.data[cy * self.width + cx] = value;
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    fn clamp_coords(&self, x: f64, y: f64) -> (usize, usize) {
        let cx = ((x / self.cell_size).max(0.0) as usize).min(self.width - 1);
        let cy = ((y / self.cell_size).max(0.0) as usize).min(self.height - 1);
        (cx, cy)
    }
}

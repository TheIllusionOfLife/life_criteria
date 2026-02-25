//! Trivial feedforward neural network: 8 inputs → 16 hidden (tanh) → 4 outputs (tanh).
//! Stack-allocated, no heap. 212 weights total.
//!
//! Inputs:  position(2) + velocity(2) + internal_state(3) + neighbor_count(1) = 8
//! Outputs: velocity_delta(2) + state_delta(2) = 4

const INPUT_SIZE: usize = 8;
const HIDDEN_SIZE: usize = 16;
const OUTPUT_SIZE: usize = 4;

#[derive(Clone, Debug)]
pub struct NeuralNet {
    // weights: input→hidden (8×16) + hidden bias (16) + hidden→output (16×4) + output bias (4)
    // Total: 128 + 16 + 64 + 4 = 212 parameters
    pub w_ih: [[f32; HIDDEN_SIZE]; INPUT_SIZE],  // 8×16
    pub b_h: [f32; HIDDEN_SIZE],                 // 16
    pub w_ho: [[f32; OUTPUT_SIZE]; HIDDEN_SIZE], // 16×4
    pub b_o: [f32; OUTPUT_SIZE],                 // 4
}

impl NeuralNet {
    /// Create a NN from an iterator of f32 values. Panics if fewer than WEIGHT_COUNT values.
    pub fn from_weights(mut weights: impl Iterator<Item = f32>) -> Self {
        let mut next = || {
            weights
                .next()
                .expect("insufficient weights: need WEIGHT_COUNT (212) elements")
        };

        let mut w_ih = [[0.0f32; HIDDEN_SIZE]; INPUT_SIZE];
        for row in &mut w_ih {
            for w in row.iter_mut() {
                *w = next();
            }
        }

        let mut b_h = [0.0f32; HIDDEN_SIZE];
        for b in &mut b_h {
            *b = next();
        }

        let mut w_ho = [[0.0f32; OUTPUT_SIZE]; HIDDEN_SIZE];
        for row in &mut w_ho {
            for w in row.iter_mut() {
                *w = next();
            }
        }

        let mut b_o = [0.0f32; OUTPUT_SIZE];
        for b in &mut b_o {
            *b = next();
        }

        Self {
            w_ih,
            b_h,
            w_ho,
            b_o,
        }
    }

    /// Forward pass. Returns [vel_dx, vel_dy, state_d0, state_d1].
    pub fn forward(&self, input: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        // Hidden layer
        let mut hidden = self.b_h;
        for (i, &x) in input.iter().enumerate() {
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += x * self.w_ih[i][j];
            }
        }
        // tanh activation
        for h in &mut hidden {
            *h = h.tanh();
        }

        // Output layer
        let mut output = self.b_o;
        for (i, &h) in hidden.iter().enumerate() {
            for (j, o) in output.iter_mut().enumerate() {
                *o += h * self.w_ho[i][j];
            }
        }
        // tanh activation
        for o in &mut output {
            *o = o.tanh();
        }

        output
    }

    /// Flatten network parameters in the same order expected by `from_weights`.
    pub fn to_weight_vec(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(Self::WEIGHT_COUNT);
        for row in &self.w_ih {
            out.extend_from_slice(row);
        }
        out.extend_from_slice(&self.b_h);
        for row in &self.w_ho {
            out.extend_from_slice(row);
        }
        out.extend_from_slice(&self.b_o);
        out
    }

    pub const WEIGHT_COUNT: usize =
        INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE;
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn weight_count_matches_dimensions() {
        assert_eq!(NeuralNet::WEIGHT_COUNT, 8 * 16 + 16 + 16 * 4 + 4);
        assert_eq!(NeuralNet::WEIGHT_COUNT, 212);
    }

    #[test]
    fn forward_output_bounded_by_tanh() {
        let nn = NeuralNet::from_weights((0..NeuralNet::WEIGHT_COUNT).map(|i| (i as f32) * 0.01));
        let input = [1.0f32; INPUT_SIZE];
        let output = nn.forward(&input);
        for &o in &output {
            assert!((-1.0..=1.0).contains(&o), "output {o} outside tanh range");
        }
    }

    #[test]
    fn zero_weights_produce_zero_output() {
        let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
        let input = [1.0f32; INPUT_SIZE];
        let output = nn.forward(&input);
        for &o in &output {
            assert!((o.abs()) < 1e-7, "expected ~0 with zero weights, got {o}");
        }
    }

    #[test]
    #[should_panic(expected = "insufficient weights")]
    fn from_weights_panics_on_short_iterator() {
        NeuralNet::from_weights(std::iter::repeat_n(0.0f32, 10));
    }

    #[test]
    fn to_weight_vec_round_trips_into_equivalent_network() {
        let nn = NeuralNet::from_weights((0..NeuralNet::WEIGHT_COUNT).map(|i| i as f32 * 0.01));
        let round_trip = NeuralNet::from_weights(nn.to_weight_vec().into_iter());
        let input = [0.25f32; INPUT_SIZE];
        assert_eq!(nn.forward(&input), round_trip.forward(&input));
    }

    proptest! {
        #[test]
        fn proptest_forward_outputs_finite_and_bounded(
            weights in proptest::collection::vec(-10.0f32..10.0f32, NeuralNet::WEIGHT_COUNT),
            inputs in proptest::collection::vec(-5.0f32..5.0f32, INPUT_SIZE),
        ) {
            let nn = NeuralNet::from_weights(weights.into_iter());
            let input: [f32; INPUT_SIZE] = inputs.try_into().expect("input size should match");
            let output = nn.forward(&input);
            prop_assert!(output.iter().all(|o| o.is_finite() && *o >= -1.0 && *o <= 1.0));
        }
    }
}

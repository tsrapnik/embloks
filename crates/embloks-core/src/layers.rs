//! Display layers (E, B, S, particles) and the controls the layers panel offers.

use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

pub const GAINS: [f64; 9] = [1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0, 2.0, 4.0, 8.0, 16.0];
pub const OPACITIES: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 1.0];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub label: String,
    pub visible: bool,
    /// Field values are multiplied by this before mapping to arrow length / colour.
    pub gain: f64,
    pub opacity: f64,
    pub log_scale: bool,
    /// Auto: the gain follows the field so its strongest ~1% just reaches full strength.
    pub auto: bool,
    /// Particles can only be shown or hidden.
    pub adjustable: bool,
}

impl Layer {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            visible: true,
            gain: 1.0,
            opacity: 1.0,
            log_scale: true,
            auto: false,
            adjustable: true,
        }
    }

    /// Manual gain step. Leaves auto mode, stepping from the gain auto had reached.
    pub fn step_gain(&mut self, steps: i32) {
        self.auto = false;
        self.gain = step(&GAINS, self.gain, steps);
    }

    pub fn gain_label(&self) -> String {
        if self.auto {
            "auto".into()
        } else if self.gain >= 1.0 {
            format!("×{}", self.gain)
        } else {
            format!("×1/{}", (1.0 / self.gain).round())
        }
    }

    pub fn step_opacity(&mut self, steps: i32) {
        self.opacity = step(&OPACITIES, self.opacity, steps);
    }

    /// Display strength 0..1 for a field magnitude, at the current gain.
    ///
    /// Log scale maps [low, high] onto [0, 1] logarithmically; linear scale maps
    /// [0, high] linearly. The gain shifts which magnitudes reach full strength.
    pub fn strength(&self, magnitude: f64, low: f64, high: f64) -> f64 {
        let scaled = magnitude * self.gain;
        let t = if self.log_scale { (scaled / low).ln() / (high / low).ln() } else { scaled / high };
        if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) }
    }

    /// Display strengths for field magnitudes; in auto mode the gain follows them first.
    pub fn intensity(&mut self, magnitudes: &[f64], low: f64, high: f64) -> Vec<f64> {
        if self.auto {
            self.follow(magnitudes, high);
        }
        magnitudes.iter().map(|&m| self.strength(m, low, high)).collect()
    }

    /// Move the gain towards high / (99th percentile of the nonzero magnitudes).
    pub fn follow(&mut self, magnitudes: &[f64], high: f64) {
        let mut nonzero: Vec<f64> = magnitudes.iter().copied().filter(|&m| m > 0.0).collect();
        if nonzero.is_empty() {
            return;
        }
        nonzero.sort_by(f64::total_cmp);
        self.ease_towards(high / percentile(&nonzero, 99.0));
    }

    /// A fraction of the way to `target` per frame, in log space, so the colours
    /// don't pump with noise.
    pub fn ease_towards(&mut self, target: f64) {
        if target.is_finite() && target > 0.0 {
            self.gain = (self.gain * (target / self.gain).powf(0.15)).clamp(1e-6, 1e6);
        }
    }

    /// Apply a click on one of the panel's buttons.
    pub fn apply(&mut self, control: LayerControl) {
        match control {
            LayerControl::Show => self.visible = !self.visible,
            LayerControl::ScaleDown => self.step_gain(-1),
            LayerControl::ScaleUp => self.step_gain(1),
            LayerControl::ScaleAuto => self.auto = !self.auto,
            LayerControl::OpacityDown => self.step_opacity(-1),
            LayerControl::OpacityUp => self.step_opacity(1),
            LayerControl::Mode => self.log_scale = !self.log_scale,
        }
    }
}

/// Linear-interpolation percentile of sorted values (numpy's default).
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    let x = q / 100.0 * (sorted.len() - 1) as f64;
    let i = x.floor() as usize;
    let j = (i + 1).min(sorted.len() - 1);
    sorted[i] + (sorted[j] - sorted[i]) * (x - i as f64)
}

fn step(values: &[f64], current: f64, steps: i32) -> f64 {
    let index = (0..values.len())
        .min_by(|&a, &b| (values[a] - current).abs().total_cmp(&(values[b] - current).abs()))
        .unwrap() as i32;
    values[(index + steps).clamp(0, values.len() as i32 - 1) as usize]
}

/// The buttons of one row of the layers panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerControl {
    Show,
    ScaleDown,
    ScaleUp,
    ScaleAuto,
    OpacityDown,
    OpacityUp,
    Mode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerId {
    E = 0,
    B = 1,
    S = 2,
    P = 3,
}

impl LayerId {
    pub const ALL: [LayerId; 4] = [LayerId::E, LayerId::B, LayerId::S, LayerId::P];
}

/// The four display layers, indexed by LayerId.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layers(pub [Layer; 4]);

impl Default for Layers {
    fn default() -> Self {
        let mut particles = Layer::new("Particles");
        particles.adjustable = false;
        Self([
            Layer::new("E field"),
            // B and S grow with the current (S with current squared), so they scale
            // themselves; a fixed scale would saturate the whole screen for strong
            // currents. B is linear: on a log scale a loop's weak return field outside
            // looks nearly as strong as inside.
            Layer { auto: true, log_scale: false, ..Layer::new("B field") },
            // Off by default: Poynting arrows sit on the same cells as the E arrows.
            Layer { visible: false, auto: true, ..Layer::new("S energy flow") },
            particles,
        ])
    }
}

impl Index<LayerId> for Layers {
    type Output = Layer;
    fn index(&self, id: LayerId) -> &Layer {
        &self.0[id as usize]
    }
}

impl IndexMut<LayerId> for Layers {
    fn index_mut(&mut self, id: LayerId) -> &mut Layer {
        &mut self.0[id as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_intensity_spans_bounds_and_gain_shifts_it() {
        let mut layer = Layer::new("E");
        let t = layer.intensity(&[0.0, 1e-3, (1e-3f64).sqrt(), 1.0, 5.0], 1e-3, 1.0);
        for (a, b) in t.iter().zip([0.0, 0.0, 0.5, 1.0, 1.0]) {
            assert!((a - b).abs() < 1e-12);
        }
        layer.gain = 10.0;
        assert!((layer.intensity(&[0.1], 1e-3, 1.0)[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_intensity_is_proportional() {
        let mut layer = Layer { log_scale: false, ..Layer::new("B") };
        assert_eq!(layer.intensity(&[0.0, 0.25, 0.5, 2.0], 1e-3, 0.5), vec![0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn gain_and_opacity_steps_are_clamped() {
        let mut layer = Layer::new("E");
        for _ in 0..20 {
            layer.step_gain(1);
            layer.step_opacity(-1);
        }
        assert_eq!((layer.gain, layer.opacity), (GAINS[8], OPACITIES[0]));
    }

    #[test]
    fn panel_buttons_change_their_layer() {
        let mut layers = Layers::default();
        for control in [LayerControl::Show, LayerControl::ScaleUp, LayerControl::OpacityDown, LayerControl::Mode] {
            layers[LayerId::E].apply(control);
        }
        layers[LayerId::P].apply(LayerControl::Show);
        let e = &layers[LayerId::E];
        assert!(!e.visible && e.gain == 2.0 && e.opacity == 0.75 && !e.log_scale);
        assert!(!layers[LayerId::P].visible && !layers[LayerId::P].adjustable);
    }

    #[test]
    fn auto_gain_settles_so_the_strongest_values_reach_full_strength() {
        let mut layer = Layer { auto: true, log_scale: false, ..Layer::new("B") };
        let magnitude: Vec<f64> = (0..1000).map(|i| 0.02 * i as f64 / 999.0).collect();
        let mut t = vec![];
        for _ in 0..200 {
            t = layer.intensity(&magnitude, 1e-3, 1.0);
        }
        let mut nonzero: Vec<f64> = t.into_iter().filter(|&v| v > 0.0).collect();
        nonzero.sort_by(f64::total_cmp);
        assert!((percentile(&nonzero, 99.0) - 1.0).abs() < 0.02);
        assert_eq!(layer.gain_label(), "auto");
    }

    #[test]
    fn stepping_the_gain_leaves_auto_from_the_reached_gain() {
        let mut layer = Layer { auto: true, ..Layer::new("B") };
        layer.gain = 7.3; // Wherever auto had got to.
        layer.step_gain(1);
        assert!(!layer.auto && layer.gain == 16.0);
        assert_eq!(layer.gain_label(), "×16");
        layer.step_gain(-8);
        assert_eq!(layer.gain_label(), "×1/16");
    }

    #[test]
    fn clicking_the_scale_value_toggles_auto() {
        let mut layer = Layer::new("E field");
        layer.apply(LayerControl::ScaleAuto);
        assert!(layer.auto);
    }
}

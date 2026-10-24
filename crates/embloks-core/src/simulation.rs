//! One tick of physics: battery feedback, electrons, then fields.

use ndarray::{Array2, Array3};

use crate::battery;
use crate::close_range::CloseRange;
use crate::maxwell::Maxwell;
use crate::physics;
use crate::world::World;

/// The physics of a World: the fields, close-range forces and batteries.
///
/// The fields start from rest: only the static field of the charges present.
#[derive(Clone)]
pub struct Simulation {
    pub world: World,
    pub maxwell: Maxwell,
    pub close_range: CloseRange,
    /// Simulated time since the scene started.
    pub time: f64,
    fields_key: (u64, u64),
}

impl Simulation {
    pub fn new(world: World) -> Self {
        let maxwell = Maxwell::for_world(&world);
        let close_range = CloseRange::new(&world.settings);
        let fields_key = world.scene_key();
        Self { world, maxwell, close_range, time: 0.0, fields_key }
    }

    /// Bring E and the potential up to date after painting, even while paused.
    ///
    /// Ticks keep Gauss's law themselves; this covers charge that appeared or
    /// vanished since (painted conductors, test charges).
    pub fn sync(&mut self) {
        if self.world.scene_key() != self.fields_key {
            let (rho, _) = self.maxwell.sources(&self.world);
            self.maxwell.enforce_gauss(&rho);
            self.fields_key = self.world.scene_key();
        }
    }

    /// Electric potential at the cell centres, close-range part included.
    ///
    /// The close-range part is only added on the cells in `where_`, if given
    /// (see CloseRange::potential).
    pub fn potential(&mut self, where_: Option<&Array2<bool>>) -> Array2<f64> {
        self.sync();
        let mut potential = self.maxwell.visible_potential();
        if self.close_range.enabled() {
            potential += &self.close_range.potential(&self.world, where_);
        }
        potential
    }

    /// E (cols, rows, 2) at the cell centres.
    pub fn e_field(&self) -> Array3<f64> {
        self.maxwell.node_fields().0
    }

    /// Advance everything by one tick of dt; returns the electron substeps.
    pub fn step(&mut self) -> usize {
        let dt = self.world.settings.dt;
        let potential = self.potential(Some(&battery::probe_cells(&self.world)));
        let current = physics::current_density(&self.world);
        let settings = self.world.settings.clone();
        battery::trim(&mut self.world, &potential, &current, dt, &settings, self.time);
        let substeps = physics::step(&mut self.world, &mut self.maxwell, &self.close_range, dt, self.time);
        self.time += dt;
        substeps
    }

    /// Run whole ticks covering `seconds` of simulated time.
    pub fn run(&mut self, seconds: f64) -> &mut Self {
        for _ in 0..(seconds / self.world.settings.dt).round() as usize {
            self.step();
        }
        self
    }
}

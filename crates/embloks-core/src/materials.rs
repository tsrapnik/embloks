use ndarray::Array2;
use serde::{Deserialize, Serialize};

pub type Cell = (usize, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Material {
    Vacuum = 0,
    Conductor = 1,
    Battery = 2,
}

impl Material {
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Material::Conductor,
            2 => Material::Battery,
            _ => Material::Vacuum,
        }
    }
}

/// Unit vector towards the + terminal for each Orientation value, for table lookups.
pub const ORIENTATION_VECTORS: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];

/// Direction of a battery's + terminal, in screen coordinates (y points down).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Orientation {
    Right = 0,
    Down = 1,
    Left = 2,
    Up = 3,
}

impl Orientation {
    pub fn from_u8(value: u8) -> Self {
        [Orientation::Right, Orientation::Down, Orientation::Left, Orientation::Up][value as usize % 4]
    }

    pub fn vector(self) -> (i32, i32) {
        let (x, y) = ORIENTATION_VECTORS[self as usize];
        (x as i32, y as i32)
    }

    /// The next orientation, clockwise on screen.
    pub fn rotated(self) -> Self {
        Self::from_u8(self as u8 + 1)
    }

    pub fn name(self) -> &'static str {
        match self {
            Orientation::Right => "right",
            Orientation::Down => "down",
            Orientation::Left => "left",
            Orientation::Up => "up",
        }
    }
}

/// Properties of the cells being painted (beyond the material itself).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintProps {
    pub resistance: f64,
    pub emf: f64,
    pub orientation: Orientation,
    pub frequency: f64,
    pub density: f64,
}

impl Default for PaintProps {
    fn default() -> Self {
        Self { resistance: 0.0, emf: 0.0, orientation: Orientation::Right, frequency: 0.0, density: 1.0 }
    }
}

/// Per-cell material properties, as arrays of shape (cols, rows) indexed [col, row].
#[derive(Clone, Debug)]
pub struct Materials {
    pub cols: usize,
    pub rows: usize,
    pub kind: Array2<u8>,
    // Conductors and batteries (a battery's internal resistance).
    pub resistance: Array2<f64>,
    // Charge density of conductive cells, in units of a wire's: ion charge and
    // electrons per cell are this many times cell_charge and carriers_per_cell.
    pub density: Array2<f64>,
    // Only meaningful for batteries.
    pub emf: Array2<f64>,
    pub orientation: Array2<u8>,
    // AC frequency (cycles per time unit); 0 is a DC battery.
    pub frequency: Array2<f64>,
    // Bumped on every change, so caches (rendering, physics) know when to rebuild.
    pub version: u64,
}

impl Materials {
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            kind: Array2::zeros((cols, rows)),
            resistance: Array2::zeros((cols, rows)),
            density: Array2::zeros((cols, rows)),
            emf: Array2::zeros((cols, rows)),
            orientation: Array2::zeros((cols, rows)),
            frequency: Array2::zeros((cols, rows)),
            version: 0,
        }
    }

    pub fn material(&self, cell: Cell) -> Material {
        Material::from_u8(self.kind[cell])
    }

    /// Cells that hold a fixed ion and mobile electrons: conductors and batteries.
    ///
    /// A battery conducts like a wire, but also pushes its electrons towards its
    /// minus side (see battery.rs).
    pub fn conductive(&self) -> Array2<bool> {
        self.kind.mapv(|k| k == Material::Conductor as u8 || k == Material::Battery as u8)
    }

    pub fn is_conductive(&self, cell: Cell) -> bool {
        self.kind[cell] != Material::Vacuum as u8
    }

    pub fn in_bounds(&self, c: i64, r: i64) -> bool {
        c >= 0 && r >= 0 && (c as usize) < self.cols && (r as usize) < self.rows
    }

    /// Set the given cells to `material`, clearing properties it doesn't use.
    pub fn paint(&mut self, cells: &[Cell], material: Material, props: PaintProps) {
        let is_battery = material == Material::Battery;
        let is_conductive = material == Material::Conductor || is_battery;
        for &cell in cells {
            self.kind[cell] = material as u8;
            self.resistance[cell] = if is_conductive { props.resistance } else { 0.0 };
            self.density[cell] = if is_conductive { props.density } else { 0.0 };
            self.emf[cell] = if is_battery { props.emf } else { 0.0 };
            self.orientation[cell] = if is_battery { props.orientation as u8 } else { 0 };
            self.frequency[cell] = if is_battery { props.frequency } else { 0.0 };
        }
        self.version += 1;
    }

    pub fn clear(&mut self) {
        let all: Vec<Cell> = self.kind.indexed_iter().map(|(cell, _)| cell).collect();
        self.paint(&all, Material::Vacuum, PaintProps::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(resistance: f64, emf: f64, orientation: Orientation) -> PaintProps {
        PaintProps { resistance, emf, orientation, ..PaintProps::default() }
    }

    #[test]
    fn new_materials_are_all_vacuum() {
        let m = Materials::new(5, 4);
        assert_eq!(m.kind.dim(), (5, 4));
        assert!(m.kind.iter().all(|&k| k == Material::Vacuum as u8));
    }

    #[test]
    fn paint_conductor_sets_resistance_only() {
        let mut m = Materials::new(5, 4);
        m.paint(&[(1, 2)], Material::Conductor, props(2.0, 9.0, Orientation::Right));
        assert_eq!(m.material((1, 2)), Material::Conductor);
        assert_eq!(m.resistance[(1, 2)], 2.0);
        assert_eq!(m.emf[(1, 2)], 0.0);
    }

    #[test]
    fn paint_battery_sets_emf_and_orientation() {
        let mut m = Materials::new(5, 4);
        m.paint(&[(3, 0)], Material::Battery, props(2.0, 7.0, Orientation::Up));
        assert_eq!(m.material((3, 0)), Material::Battery);
        assert_eq!((m.emf[(3, 0)], m.orientation[(3, 0)], m.resistance[(3, 0)]), (7.0, Orientation::Up as u8, 2.0));
        assert!(m.conductive()[(3, 0)]);
    }

    #[test]
    fn painting_vacuum_clears_properties() {
        let mut m = Materials::new(5, 4);
        let cells = [(0, 0), (1, 0)];
        m.paint(&cells, Material::Battery, props(0.0, 7.0, Orientation::Left));
        m.paint(&cells, Material::Vacuum, PaintProps::default());
        assert!(m.kind.iter().all(|&k| k == 0));
        assert!(m.emf.iter().all(|&e| e == 0.0));
        assert!(m.orientation.iter().all(|&o| o == 0));
    }

    #[test]
    fn paint_and_clear_bump_version() {
        let mut m = Materials::new(5, 4);
        let v = m.version;
        m.paint(&[(0, 0)], Material::Conductor, props(1.0, 0.0, Orientation::Right));
        assert!(m.version > v);
        let v = m.version;
        m.clear();
        assert!(m.version > v);
        assert!(m.kind.iter().all(|&k| k == Material::Vacuum as u8));
    }

    #[test]
    fn orientation_rotates_clockwise_on_screen() {
        let mut o = Orientation::Right;
        let mut seen = vec![];
        for _ in 0..4 {
            seen.push(o.vector());
            o = o.rotated();
        }
        // Screen y points down, so right -> down -> left -> up is clockwise.
        assert_eq!(seen, vec![(1, 0), (0, 1), (-1, 0), (0, -1)]);
        assert_eq!(o, Orientation::Right);
    }
}

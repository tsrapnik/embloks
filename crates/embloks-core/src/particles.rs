/// Mobile charges (electrons), stored as parallel arrays.
///
/// Positions are in cell units: cell (col, row) spans [col, col + 1) x [row, row + 1),
/// with y pointing down like the screen. Fixed ions are not stored here; every
/// conductive cell implicitly has one spread over it (see World).
#[derive(Clone, Debug, Default)]
pub struct Particles {
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    pub charge: Vec<f64>,
    pub mass: Vec<f64>,
}

impl Particles {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.charge.len()
    }

    pub fn is_empty(&self) -> bool {
        self.charge.is_empty()
    }

    pub fn add(&mut self, pos: &[[f64; 2]], charge: f64, mass: f64) {
        self.pos.extend_from_slice(pos);
        self.vel.extend(std::iter::repeat_n([0.0; 2], pos.len()));
        self.charge.extend(std::iter::repeat_n(charge, pos.len()));
        self.mass.extend(std::iter::repeat_n(mass, pos.len()));
    }

    /// Drop every particle where `mask` is false.
    pub fn keep(&mut self, mask: &[bool]) {
        fn retain<T: Copy>(values: &mut Vec<T>, mask: &[bool]) {
            let mut i = 0;
            values.retain(|_| {
                i += 1;
                mask[i - 1]
            });
        }
        retain(&mut self.pos, mask);
        retain(&mut self.vel, mask);
        retain(&mut self.charge, mask);
        retain(&mut self.mass, mask);
    }

    /// Column and row of the cell particle `i` is in.
    pub fn cell(&self, i: usize) -> (i64, i64) {
        cell_of(self.pos[i])
    }
}

/// The cell a position lies in (may be off the grid for stray positions).
pub fn cell_of(pos: [f64; 2]) -> (i64, i64) {
    (pos[0].floor() as i64, pos[1].floor() as i64)
}

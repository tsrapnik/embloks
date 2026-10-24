use crate::materials::{Cell, Material, Orientation};
use crate::settings::Settings;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Conductor,
    Battery,
    Charge,
}

impl Tool {
    /// The material this tool paints, or None for the test charge tool.
    pub fn material(self) -> Option<Material> {
        match self {
            Tool::Conductor => Some(Material::Conductor),
            Tool::Battery => Some(Material::Battery),
            Tool::Charge => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Tool::Conductor => "conductor",
            Tool::Battery => "battery",
            Tool::Charge => "test charge",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Brush {
    settings: Settings,
    pub tool: Tool,
    /// 0 paints a single cell; r paints a disc of radius r cells.
    pub radius: i32,
    pub orientation: Orientation,
    // Indices into the settings' value ladders.
    pub resistance_index: usize,
    pub emf_index: usize,
    pub charge_index: usize,
    pub frequency_index: usize,
}

fn index_of(values: &[f64], value: f64) -> usize {
    values.iter().position(|&v| v == value).unwrap_or(0)
}

impl Brush {
    pub fn new(settings: &Settings) -> Self {
        Self {
            resistance_index: index_of(&settings.resistance_values, settings.default_resistance),
            emf_index: index_of(&settings.emf_values, settings.default_emf),
            charge_index: index_of(&settings.test_charge_values, settings.default_test_charge),
            frequency_index: 0,
            tool: Tool::Conductor,
            radius: 0,
            orientation: Orientation::Right,
            settings: settings.clone(),
        }
    }

    pub fn resistance(&self) -> f64 {
        self.settings.resistance_values[self.resistance_index]
    }

    pub fn emf(&self) -> f64 {
        self.settings.emf_values[self.emf_index]
    }

    /// Battery frequency: 0 is DC, otherwise AC cycles per time unit.
    pub fn frequency(&self) -> f64 {
        self.settings.ac_frequencies[self.frequency_index]
    }

    pub fn step_frequency(&mut self) {
        self.frequency_index = (self.frequency_index + 1) % self.settings.ac_frequencies.len();
    }

    pub fn charge(&self) -> f64 {
        self.settings.test_charge_values[self.charge_index]
    }

    pub fn resize(&mut self, delta: i32) {
        self.radius = (self.radius + delta).clamp(0, self.settings.max_brush_radius);
    }

    pub fn rotate(&mut self) {
        self.orientation = self.orientation.rotated();
    }

    /// Step the current tool's value (resistance, EMF or charge) up or down its ladder.
    pub fn adjust_value(&mut self, delta: i32) {
        let s = &self.settings;
        let (index, len) = match self.tool {
            Tool::Conductor => (&mut self.resistance_index, s.resistance_values.len()),
            Tool::Battery => (&mut self.emf_index, s.emf_values.len()),
            Tool::Charge => (&mut self.charge_index, s.test_charge_values.len()),
        };
        *index = (*index as i32 + delta).clamp(0, len as i32 - 1) as usize;
    }

    /// The cells covered by the brush at `center`, clipped to a cols x rows grid.
    pub fn footprint(&self, center: Cell, cols: usize, rows: usize) -> Vec<Cell> {
        let r = self.radius;
        let mut cells = Vec::new();
        for dc in -r..=r {
            for dr in -r..=r {
                // (r + 0.5)^2 rounds the disc nicely: r=1 gives a full 3x3 block.
                if (dc * dc + dr * dr) as f64 > (r as f64 + 0.5).powi(2) {
                    continue;
                }
                let (c, rr) = (center.0 as i32 + dc, center.1 as i32 + dr);
                if c >= 0 && rr >= 0 && (c as usize) < cols && (rr as usize) < rows {
                    cells.push((c as usize, rr as usize));
                }
            }
        }
        cells
    }
}

/// Cells from start to end, inclusive, where consecutive cells share an edge.
///
/// Unlike Bresenham, this never steps diagonally, so a dragged one-cell-wide
/// wire is always 4-connected (diagonal contact doesn't conduct).
pub fn line_cells(start: Cell, end: Cell) -> Vec<Cell> {
    let (mut x, mut y) = (start.0 as i64, start.1 as i64);
    let (x1, y1) = (end.0 as i64, end.1 as i64);
    let (dx, dy) = ((x1 - x).abs(), (y1 - y).abs());
    let sx = if x1 > x { 1 } else { -1 };
    let sy = if y1 > y { 1 } else { -1 };
    let mut cells = vec![start];
    let (mut ix, mut iy) = (0, 0);
    while ix < dx || iy < dy {
        // Step along whichever axis the ideal line crosses a cell border on first.
        if (1 + 2 * ix) * dy < (1 + 2 * iy) * dx {
            x += sx;
            ix += 1;
        } else {
            y += sy;
            iy += 1;
        }
        cells.push((x as usize, y as usize));
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn cells_of(cells: Vec<Cell>) -> HashSet<Cell> {
        cells.into_iter().collect()
    }

    #[test]
    fn radius_zero_paints_one_cell() {
        let brush = Brush::new(&Settings::default());
        assert_eq!(cells_of(brush.footprint((5, 5), 20, 10)), HashSet::from([(5, 5)]));
    }

    #[test]
    fn radius_one_paints_3x3_block() {
        let mut brush = Brush::new(&Settings::default());
        brush.radius = 1;
        let expected: HashSet<Cell> = (4..7).flat_map(|c| (4..7).map(move |r| (c, r))).collect();
        assert_eq!(cells_of(brush.footprint((5, 5), 20, 10)), expected);
    }

    #[test]
    fn footprint_is_clipped_to_grid() {
        let mut brush = Brush::new(&Settings::default());
        brush.radius = 2;
        let cells = brush.footprint((0, 0), 20, 10);
        assert!(cells.contains(&(0, 0)));
        assert!(cells.iter().all(|&(c, r)| c < 20 && r < 10));
    }

    #[test]
    fn resize_is_clamped() {
        let s = Settings::default();
        let mut brush = Brush::new(&s);
        brush.resize(-5);
        assert_eq!(brush.radius, 0);
        brush.resize(1000);
        assert_eq!(brush.radius, s.max_brush_radius);
    }

    #[test]
    fn adjust_value_steps_the_current_tools_ladder() {
        let s = Settings::default();
        let mut brush = Brush::new(&s);
        assert_eq!(brush.resistance(), s.default_resistance);
        brush.adjust_value(1);
        assert!(brush.resistance() > s.default_resistance);
        assert_eq!(brush.emf(), s.default_emf);
        brush.tool = Tool::Battery;
        brush.adjust_value(-100);
        assert_eq!(brush.emf(), s.emf_values[0]);
    }

    #[test]
    fn line_cells_is_4_connected_and_hits_both_ends() {
        for (start, end) in [((0, 0), (0, 0)), ((0, 0), (7, 0)), ((3, 9), (3, 1)), ((0, 0), (5, 5)), ((9, 2), (1, 6))] {
            let cells = line_cells(start, end);
            assert_eq!((cells[0], *cells.last().unwrap()), (start, end));
            for pair in cells.windows(2) {
                // Neighbours share an edge: exactly one coordinate changes by one.
                let step = (pair[1].0 as i64 - pair[0].0 as i64).abs() + (pair[1].1 as i64 - pair[0].1 as i64).abs();
                assert_eq!(step, 1);
            }
        }
    }

    #[test]
    fn line_cells_stays_close_to_the_ideal_line() {
        for (x, y) in line_cells((0, 0), (10, 4)) {
            assert!((y as f64 - x as f64 * 4.0 / 10.0).abs() <= 1.0);
        }
    }
}

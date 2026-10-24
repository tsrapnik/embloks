/// A grid of square cells laid out on the screen.
///
/// Cells are addressed as (col, row), with (0, 0) top-left. The grid is
/// centered in the canvas; leftover pixels form a margin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub cols: usize,
    pub rows: usize,
    pub cell_px: i32,
    pub origin_x: i32,
    pub origin_y: i32,
}

impl Grid {
    /// Build a grid of roughly `density` columns filling a width x height area.
    ///
    /// Cells are a whole number of pixels, so the actual column count can be
    /// higher than requested (e.g. 150 -> 6 px cells -> 166 columns at 1000 px).
    pub fn fit(density: u32, width: i32, height: i32) -> Self {
        let cell_px = (width / density.max(1) as i32).max(1);
        let cols = (width / cell_px).max(1);
        let rows = (height / cell_px).max(1);
        Self {
            cols: cols as usize,
            rows: rows as usize,
            cell_px,
            origin_x: (width - cols * cell_px) / 2,
            origin_y: (height - rows * cell_px) / 2,
        }
    }

    /// The largest grid of exactly cols x rows cells that fits in width x height.
    pub fn for_shape(cols: usize, rows: usize, width: i32, height: i32) -> Self {
        let cell_px = (width / cols as i32).min(height / rows as i32).max(1);
        Self {
            cols,
            rows,
            cell_px,
            origin_x: (width - cols as i32 * cell_px) / 2,
            origin_y: (height - rows as i32 * cell_px) / 2,
        }
    }

    pub fn pixel_width(&self) -> i32 {
        self.cols as i32 * self.cell_px
    }

    pub fn pixel_height(&self) -> i32 {
        self.rows as i32 * self.cell_px
    }

    /// The cell under screen point (x, y), or None outside the grid.
    pub fn cell_at(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let col = ((x - self.origin_x as f64) / self.cell_px as f64).floor();
        let row = ((y - self.origin_y as f64) / self.cell_px as f64).floor();
        let inside = col >= 0.0 && row >= 0.0 && (col as usize) < self.cols && (row as usize) < self.rows;
        inside.then_some((col as usize, row as usize))
    }

    /// Screen rectangle (x, y, w, h) of a cell.
    pub fn cell_rect(&self, col: usize, row: usize) -> (i32, i32, i32, i32) {
        (
            self.origin_x + col as i32 * self.cell_px,
            self.origin_y + row as i32 * self.cell_px,
            self.cell_px,
            self.cell_px,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_uses_requested_columns_when_they_divide_evenly() {
        let grid = Grid::fit(40, 1000, 800);
        assert_eq!((grid.cols, grid.rows, grid.cell_px), (40, 32, 25));
        assert_eq!((grid.origin_x, grid.origin_y), (0, 0));
    }

    #[test]
    fn fit_centers_grid_and_stays_inside_window() {
        let grid = Grid::fit(33, 1000, 800);
        assert!(grid.pixel_width() <= 1000 && grid.pixel_height() <= 800);
        assert_eq!(grid.origin_x, (1000 - grid.pixel_width()) / 2);
        assert_eq!(grid.origin_y, (800 - grid.pixel_height()) / 2);
    }

    #[test]
    fn cell_at_finds_cell_from_any_point_inside_its_rect() {
        let grid = Grid::fit(40, 1000, 800);
        for (col, row) in [(0, 0), (5, 7), (39, 31)] {
            let (x, y, w, h) = grid.cell_rect(col, row);
            let (x, y, w, h) = (x as f64, y as f64, w as f64, h as f64);
            assert_eq!(grid.cell_at(x, y), Some((col, row)));
            assert_eq!(grid.cell_at(x + w - 1.0, y + h - 1.0), Some((col, row)));
            assert_eq!(grid.cell_at(x + w / 2.0, y + h / 2.0), Some((col, row)));
        }
    }

    #[test]
    fn cell_at_returns_none_outside_grid() {
        let grid = Grid::fit(40, 1000, 800);
        for (x, y) in [(-1.0, 10.0), (10.0, -1.0), (1000.0, 10.0), (10.0, 800.0)] {
            assert_eq!(grid.cell_at(x, y), None);
        }
    }

    #[test]
    fn cell_at_returns_none_in_centering_margin() {
        let grid = Grid::fit(33, 1000, 800);
        assert!(grid.origin_x > 0);
        assert_eq!(grid.cell_at((grid.origin_x - 1) as f64, grid.origin_y as f64), None);
    }
}

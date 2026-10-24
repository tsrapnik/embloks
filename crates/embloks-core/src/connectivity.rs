use ndarray::{Array2, s};

use crate::materials::Materials;

/// Which neighbouring cells conduct into each other, and the resulting islands.
///
/// Conductive cells (conductors and batteries) connect only through shared
/// edges (never corners).
#[derive(Clone, Debug)]
pub struct Connectivity {
    /// open_x[c, r]: cells (c, r) and (c + 1, r) connect. Shape (cols - 1, rows).
    pub open_x: Array2<bool>,
    /// open_y[c, r]: cells (c, r) and (c, r + 1) connect. Shape (cols, rows - 1).
    pub open_y: Array2<bool>,
    /// Island id per cell, 0..count-1, or -1 for non-conductors. Shape (cols, rows).
    pub labels: Array2<i64>,
    pub count: usize,
}

pub fn compute_connectivity(materials: &Materials) -> Connectivity {
    let conductive = materials.conductive();
    let (cols, rows) = conductive.dim();
    let open_x = &conductive.slice(s![..cols.saturating_sub(1), ..]) & &conductive.slice(s![1.., ..]);
    let open_y = &conductive.slice(s![.., ..rows.saturating_sub(1)]) & &conductive.slice(s![.., 1..]);

    // Flood fill in scan order (column-major, as the cells are indexed), so islands
    // are numbered by their first cell like a connected-components labelling.
    let mut labels = Array2::from_elem((cols, rows), -1i64);
    let mut count = 0;
    let mut stack = Vec::new();
    for c in 0..cols {
        for r in 0..rows {
            if !conductive[(c, r)] || labels[(c, r)] >= 0 {
                continue;
            }
            labels[(c, r)] = count;
            stack.push((c, r));
            while let Some((x, y)) = stack.pop() {
                let mut visit = |cell: (usize, usize), open: bool| {
                    if open && labels[cell] < 0 {
                        labels[cell] = count;
                        stack.push(cell);
                    }
                };
                if x > 0 {
                    visit((x - 1, y), open_x[(x - 1, y)]);
                }
                if x + 1 < cols {
                    visit((x + 1, y), open_x[(x, y)]);
                }
                if y > 0 {
                    visit((x, y - 1), open_y[(x, y - 1)]);
                }
                if y + 1 < rows {
                    visit((x, y + 1), open_y[(x, y)]);
                }
            }
            count += 1;
        }
    }
    Connectivity { open_x, open_y, labels, count: count as usize }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{Material, Orientation, PaintProps};

    fn paint(m: &mut Materials, cells: &[(usize, usize)], material: Material, orientation: Orientation) {
        m.paint(cells, material, PaintProps { resistance: 1.0, orientation, ..PaintProps::default() });
    }

    #[test]
    fn vacuum_has_no_islands() {
        let c = compute_connectivity(&Materials::new(4, 3));
        assert_eq!(c.count, 0);
        assert!(c.labels.iter().all(|&l| l == -1));
    }

    #[test]
    fn edge_neighbours_form_one_island() {
        let mut m = Materials::new(5, 5);
        paint(&mut m, &[(1, 1), (2, 1), (2, 2)], Material::Conductor, Orientation::Right);
        let c = compute_connectivity(&m);
        assert_eq!(c.count, 1);
        assert!([(1, 1), (2, 1), (2, 2)].iter().all(|&cell| c.labels[cell] == 0));
    }

    #[test]
    fn diagonal_contact_does_not_connect() {
        let mut m = Materials::new(5, 5);
        paint(&mut m, &[(1, 1), (2, 2)], Material::Conductor, Orientation::Right);
        let c = compute_connectivity(&m);
        assert_eq!(c.count, 2);
        assert_ne!(c.labels[(1, 1)], c.labels[(2, 2)]);
    }

    #[test]
    fn battery_conducts_on_all_sides() {
        let mut m = Materials::new(5, 5);
        paint(&mut m, &[(2, 2)], Material::Battery, Orientation::Right);
        paint(&mut m, &[(1, 2), (3, 2), (2, 1)], Material::Conductor, Orientation::Right);
        let c = compute_connectivity(&m);
        assert_eq!(c.count, 1);
        assert!([(1, 2), (3, 2), (2, 1)].iter().all(|&cell| c.labels[cell] == c.labels[(2, 2)]));
        assert!(c.open_x[(1, 2)] && c.open_x[(2, 2)] && c.open_y[(2, 1)]);
    }

    #[test]
    fn open_arrays_have_face_shapes() {
        let c = compute_connectivity(&Materials::new(6, 4));
        assert_eq!(c.open_x.dim(), (5, 4));
        assert_eq!(c.open_y.dim(), (6, 3));
    }
}

use crate::vec3::*;
use serde::{Deserialize, Serialize};

/// Planar opening of a duct. `normal` points out of the duct material (away from the channel).
/// The channel lies on the `-normal` side of the cap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortCap {
    pub origin: V3,
    pub normal: V3,
    /// Boundary loop of the opening (mm), roughly in the cap plane.
    pub polygon: Vec<V3>,
}

impl PortCap {
    pub fn basis(&self) -> (V3, V3) {
        basis(self.normal)
    }

    /// Polygon projected to the plane basis.
    pub fn polygon_2d(&self) -> Vec<[f64; 2]> {
        let (u, v) = self.basis();
        self.polygon.iter().map(|p| { let d = sub(*p, self.origin); [dot(d, u), dot(d, v)] }).collect()
    }

    pub fn area(&self) -> f64 {
        let p = self.polygon_2d();
        let mut a = 0.0;
        for i in 0..p.len() {
            let j = (i + 1) % p.len();
            a += p[i][0] * p[j][1] - p[j][0] * p[i][1];
        }
        a.abs() * 0.5
    }

    pub fn centroid(&self) -> V3 {
        let n = self.polygon.len().max(1) as f64;
        let mut c = [0.0; 3];
        for p in &self.polygon {
            c = add(c, *p);
        }
        scale(c, 1.0 / n)
    }

    /// Signed distance of `p` along the outward normal.
    pub fn height(&self, p: V3) -> f64 {
        dot(sub(p, self.origin), self.normal)
    }

    /// Even-odd test of the in-plane projection of `p`.
    pub fn contains_projected(&self, poly2d: &[[f64; 2]], p: V3) -> bool {
        let (u, v) = self.basis();
        let d = sub(p, self.origin);
        point_in_polygon(poly2d, [dot(d, u), dot(d, v)])
    }
}

pub fn point_in_polygon(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

use crate::vec3::*;
use serde::{Deserialize, Serialize};

/// Planar opening of a duct. `normal` points out of the duct material (away from the channel).
/// The channel lies on the `-normal` side of the cap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortCap {
    pub origin: V3,
    pub normal: V3,
    /// Boundary loop of the opening (mm). In the cap plane for planar openings; a 3D curve otherwise.
    pub polygon: Vec<V3>,
    /// For non-planar openings: a triangulated surface spanning `polygon`, each triangle wound so its
    /// normal points out of the duct. Empty for planar openings.
    #[serde(default)]
    pub triangles: Vec<[V3; 3]>,
    /// Loops of islands inside a planar opening (a hub or boss the air cannot pass through). They are excluded
    /// from the opening: area, port cells and streamline seeds all skip them.
    #[serde(default)]
    pub holes: Vec<Vec<V3>>,
}

impl PortCap {
    pub fn basis(&self) -> (V3, V3) {
        basis(self.normal)
    }

    /// Polygon projected to the plane basis.
    /// Projection of `p` into the cap plane.
    pub fn project2d(&self, p: V3) -> [f64; 2] {
        let (u, v) = self.basis();
        let d = sub(p, self.origin);
        [dot(d, u), dot(d, v)]
    }

    /// The island loops projected into the cap plane.
    pub fn holes_2d(&self) -> Vec<Vec<[f64; 2]>> {
        self.holes.iter().map(|h| h.iter().map(|p| self.project2d(*p)).collect()).collect()
    }

    /// True when `p` (projected into the plane) lies inside the opening and outside every island.
    pub fn contains_in_opening(&self, poly2d: &[[f64; 2]], holes2d: &[Vec<[f64; 2]>], p: V3) -> bool {
        let q = self.project2d(p);
        point_in_polygon(poly2d, q) && !holes2d.iter().any(|h| point_in_polygon(h, q))
    }

    pub fn polygon_2d(&self) -> Vec<[f64; 2]> {
        let (u, v) = self.basis();
        self.polygon.iter().map(|p| { let d = sub(*p, self.origin); [dot(d, u), dot(d, v)] }).collect()
    }

    /// True for curved/skewed openings described by a surface instead of a plane.
    pub fn is_surface(&self) -> bool {
        !self.triangles.is_empty()
    }

    pub fn area(&self) -> f64 {
        if self.is_surface() {
            return self.triangles.iter().map(|t| 0.5 * len(cross(sub(t[1], t[0]), sub(t[2], t[0])))).sum();
        }
        let shoelace = |p: &[[f64; 2]]| {
            let mut a = 0.0;
            for i in 0..p.len() {
                let j = (i + 1) % p.len();
                a += p[i][0] * p[j][1] - p[j][0] * p[i][1];
            }
            a.abs() * 0.5
        };
        let outer = shoelace(&self.polygon_2d());
        let islands: f64 = self.holes_2d().iter().map(|h| shoelace(h)).sum();
        (outer - islands).max(0.0)
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

use crate::vec3::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: V3,
    pub max: V3,
}

impl Aabb {
    pub fn empty() -> Self {
        Aabb { min: [f64::INFINITY; 3], max: [f64::NEG_INFINITY; 3] }
    }
    pub fn grow(&mut self, p: V3) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(p[i]);
            self.max[i] = self.max[i].max(p[i]);
        }
    }
    pub fn union(&mut self, o: &Aabb) {
        self.grow(o.min);
        self.grow(o.max);
    }
    pub fn expanded(&self, m: f64) -> Aabb {
        Aabb { min: sub(self.min, [m; 3]), max: add(self.max, [m; 3]) }
    }
    pub fn is_valid(&self) -> bool {
        (0..3).all(|i| self.min[i] <= self.max[i])
    }
}

/// Indexed triangle mesh. Units are millimetres. Triangles wind counter-clockwise seen from
/// outside (normals point out of the material).
#[derive(Clone, Debug, Default)]
pub struct TriMesh {
    pub vertices: Vec<V3>,
    pub triangles: Vec<[u32; 3]>,
}

impl TriMesh {
    pub fn bounds(&self) -> Aabb {
        let mut b = Aabb::empty();
        for v in &self.vertices {
            b.grow(*v);
        }
        b
    }

    pub fn tri(&self, i: usize) -> [V3; 3] {
        let t = self.triangles[i];
        [self.vertices[t[0] as usize], self.vertices[t[1] as usize], self.vertices[t[2] as usize]]
    }

    /// Signed volume; positive for outward-wound closed meshes.
    pub fn signed_volume(&self) -> f64 {
        (0..self.triangles.len())
            .map(|i| {
                let [a, b, c] = self.tri(i);
                dot(a, cross(b, c)) / 6.0
            })
            .sum()
    }

    pub fn append(&mut self, other: &TriMesh) {
        let off = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.triangles.extend(other.triangles.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
    }

    /// Merge vertices closer than `tol` (grid quantisation) and drop degenerate triangles.
    pub fn welded(&self, tol: f64) -> TriMesh {
        let q = |p: V3| [(p[0] / tol).round() as i64, (p[1] / tol).round() as i64, (p[2] / tol).round() as i64];
        let mut map: HashMap<[i64; 3], u32> = HashMap::new();
        let mut out = TriMesh::default();
        let mut remap = Vec::with_capacity(self.vertices.len());
        for v in &self.vertices {
            let k = q(*v);
            let id = *map.entry(k).or_insert_with(|| {
                out.vertices.push(*v);
                (out.vertices.len() - 1) as u32
            });
            remap.push(id);
        }
        for t in &self.triangles {
            let n = [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]];
            if n[0] != n[1] && n[1] != n[2] && n[0] != n[2] {
                out.triangles.push(n);
            }
        }
        out
    }

    /// Rotate about the Z axis through the origin (radians).
    pub fn rotate_z(&mut self, angle: f64) {
        let (s, c) = angle.sin_cos();
        for v in &mut self.vertices {
            *v = [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]];
        }
    }

    pub fn translate(&mut self, d: V3) {
        for v in &mut self.vertices {
            *v = add(*v, d);
        }
    }
}

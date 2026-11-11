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

    /// Make the winding of every triangle consistent and outward-facing. Faces of a STEP part (or an STL) can
    /// come out flipped: they are then back-face culled and the part looks see-through, or lit from the wrong side.
    /// Neighbouring triangles must traverse their shared edge in opposite directions; each connected piece is then
    /// flipped as a whole if its volume is negative. Returns how many triangles were flipped.
    pub fn orient_outward(&mut self) -> usize {
        let nt = self.triangles.len();
        if nt == 0 {
            return 0;
        }
        // canonical vertex ids by position, so faces that were meshed separately still share edges
        let tol = 1e-5;
        let q = |p: V3| [(p[0] / tol).round() as i64, (p[1] / tol).round() as i64, (p[2] / tol).round() as i64];
        let mut ids: HashMap<[i64; 3], u32> = HashMap::new();
        let canon: Vec<u32> = self.vertices.iter().map(|v| { let n = ids.len() as u32; *ids.entry(q(*v)).or_insert(n) }).collect();
        let tri_c = |t: usize, flipped: bool| -> [u32; 3] {
            let v = self.triangles[t];
            let c = [canon[v[0] as usize], canon[v[1] as usize], canon[v[2] as usize]];
            if flipped { [c[0], c[2], c[1]] } else { c }
        };
        let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for t in 0..nt {
            let c = tri_c(t, false);
            if c[0] == c[1] || c[1] == c[2] || c[0] == c[2] { continue; }
            for k in 0..3 {
                let (a, b) = (c[k], c[(k + 1) % 3]);
                edges.entry((a.min(b), a.max(b))).or_default().push(t);
            }
        }
        let mut flip = vec![false; nt];
        let mut seen = vec![false; nt];
        let mut flipped_total = 0;
        for start in 0..nt {
            if seen[start] { continue; }
            let mut comp = vec![start];
            seen[start] = true;
            let mut i = 0;
            while i < comp.len() {
                let t = comp[i];
                i += 1;
                let c = tri_c(t, flip[t]);
                for k in 0..3 {
                    let (a, b) = (c[k], c[(k + 1) % 3]);
                    let Some(nb) = edges.get(&(a.min(b), a.max(b))) else { continue };
                    if nb.len() != 2 { continue; } // boundary or non-manifold: leave alone
                    for &t2 in nb {
                        if t2 == t || seen[t2] { continue; }
                        let c2 = tri_c(t2, flip[t2]);
                        // a consistent neighbour runs the shared edge b -> a
                        let same_dir = (0..3).any(|j| c2[j] == a && c2[(j + 1) % 3] == b);
                        if same_dir { flip[t2] = !flip[t2]; }
                        seen[t2] = true;
                        comp.push(t2);
                    }
                }
            }
            // orient the whole piece outward (positive volume)
            let vol: f64 = comp.iter().map(|&t| {
                let [a, b, c] = self.tri(t);
                let s = if flip[t] { -1.0 } else { 1.0 };
                s * dot(a, cross(b, c)) / 6.0
            }).sum();
            if vol < 0.0 {
                for &t in &comp { flip[t] = !flip[t]; }
            }
        }
        for t in 0..nt {
            if flip[t] {
                self.triangles[t].swap(1, 2);
                flipped_total += 1;
            }
        }
        flipped_total
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

//! Closed test/demo solids.
use crate::mesh::TriMesh;
use crate::vec3::*;

fn quad(m: &mut TriMesh, a: u32, b: u32, c: u32, d: u32, want: V3) {
    let n = cross(sub(m.vertices[b as usize], m.vertices[a as usize]), sub(m.vertices[c as usize], m.vertices[a as usize]));
    if dot(n, want) >= 0.0 {
        m.triangles.push([a, b, c]);
        m.triangles.push([a, c, d]);
    } else {
        m.triangles.push([a, c, b]);
        m.triangles.push([a, d, c]);
    }
}

/// Axis-aligned solid box.
pub fn solid_box(min: V3, max: V3) -> TriMesh {
    let mut m = TriMesh::default();
    for i in 0..8 {
        m.vertices.push([
            if i & 1 == 0 { min[0] } else { max[0] },
            if i & 2 == 0 { min[1] } else { max[1] },
            if i & 4 == 0 { min[2] } else { max[2] },
        ]);
    }
    let faces: [([u32; 4], V3); 6] = [
        ([0, 2, 6, 4], [-1.0, 0.0, 0.0]),
        ([1, 3, 7, 5], [1.0, 0.0, 0.0]),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
        ([2, 3, 7, 6], [0.0, 1.0, 0.0]),
        ([0, 1, 3, 2], [0.0, 0.0, -1.0]),
        ([4, 5, 7, 6], [0.0, 0.0, 1.0]),
    ];
    for (q, n) in faces {
        quad(&mut m, q[0], q[1], q[2], q[3], n);
    }
    m
}

/// Square hollow tube along +X from x=0 to `length`, centred on the X axis.
/// `outer` and `inner` are full side lengths (mm). A closed solid with a through channel.
pub fn hollow_tube(outer: f64, inner: f64, length: f64) -> TriMesh {
    let mut m = TriMesh::default();
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    for e in 0..2 {
        for (r, size) in [outer, inner].iter().enumerate() {
            let h = size * 0.5;
            for c in corners {
                let _ = r;
                m.vertices.push([e as f64 * length, c.0 * h, c.1 * h]);
            }
        }
    }
    let idx = |e: u32, r: u32, c: u32| e * 8 + r * 4 + (c % 4);
    for c in 0..4u32 {
        let mid = [(corners[c as usize].0 + corners[(c as usize + 1) % 4].0) * 0.5, (corners[c as usize].1 + corners[(c as usize + 1) % 4].1) * 0.5];
        let radial = [0.0, mid[0], mid[1]];
        quad(&mut m, idx(0, 0, c), idx(0, 0, c + 1), idx(1, 0, c + 1), idx(1, 0, c), radial);
        quad(&mut m, idx(0, 1, c), idx(0, 1, c + 1), idx(1, 1, c + 1), idx(1, 1, c), scale(radial, -1.0));
        quad(&mut m, idx(0, 0, c), idx(0, 0, c + 1), idx(0, 1, c + 1), idx(0, 1, c), [-1.0, 0.0, 0.0]);
        quad(&mut m, idx(1, 0, c), idx(1, 0, c + 1), idx(1, 1, c + 1), idx(1, 1, c), [1.0, 0.0, 0.0]);
    }
    m
}

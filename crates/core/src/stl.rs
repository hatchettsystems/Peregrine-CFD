//! STL reader (binary and ASCII). Vertices are welded on load.
use crate::mesh::TriMesh;

#[derive(Debug, thiserror::Error)]
pub enum StlError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed STL: {0}")]
    Malformed(String),
}

pub fn read_stl(path: &std::path::Path) -> Result<TriMesh, StlError> {
    parse_stl(&std::fs::read(path)?)
}

pub fn parse_stl(bytes: &[u8]) -> Result<TriMesh, StlError> {
    let mut raw = TriMesh::default();
    let binary_len = if bytes.len() >= 84 {
        let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        (bytes.len() == 84 + 50 * n).then_some(n)
    } else {
        None
    };
    if let Some(n) = binary_len {
        for i in 0..n {
            let base = 84 + i * 50 + 12;
            let mut t = [0u32; 3];
            for (k, slot) in t.iter_mut().enumerate() {
                let o = base + k * 12;
                let f = |j: usize| f32::from_le_bytes([bytes[o + j * 4], bytes[o + j * 4 + 1], bytes[o + j * 4 + 2], bytes[o + j * 4 + 3]]) as f64;
                raw.vertices.push([f(0), f(1), f(2)]);
                *slot = (raw.vertices.len() - 1) as u32;
            }
            raw.triangles.push(t);
        }
    } else {
        let text = std::str::from_utf8(bytes).map_err(|e| StlError::Malformed(e.to_string()))?;
        let mut cur = vec![];
        for line in text.lines() {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("vertex") {
                let v: Vec<f64> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                if v.len() != 3 {
                    return Err(StlError::Malformed(format!("bad vertex line: {l}")));
                }
                raw.vertices.push([v[0], v[1], v[2]]);
                cur.push((raw.vertices.len() - 1) as u32);
                if cur.len() == 3 {
                    raw.triangles.push([cur[0], cur[1], cur[2]]);
                    cur.clear();
                }
            }
        }
        if raw.triangles.is_empty() {
            return Err(StlError::Malformed("no facets".into()));
        }
    }
    Ok(raw.welded(1e-5))
}

pub fn write_binary_stl(mesh: &TriMesh, path: &std::path::Path) -> std::io::Result<()> {
    use crate::vec3::*;
    let mut out = vec![0u8; 80];
    out.extend((mesh.triangles.len() as u32).to_le_bytes());
    for i in 0..mesh.triangles.len() {
        let [a, b, c] = mesh.tri(i);
        let n = normalize(cross(sub(b, a), sub(c, a)));
        for v in [n, a, b, c] {
            for x in v {
                out.extend((x as f32).to_le_bytes());
            }
        }
        out.extend([0u8, 0]);
    }
    std::fs::write(path, out)
}

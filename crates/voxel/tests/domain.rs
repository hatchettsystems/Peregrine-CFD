use peregrine_core::{shapes, vec3::*, find_port_candidates};
use peregrine_voxel::*;

#[test]
fn tube_domain() {
    let tube = shapes::hollow_tube(8.0, 4.0, 30.0);
    let caps = find_port_candidates(&tube, 1.0);
    let ports: Vec<PortSpec> = caps.iter().map(|c| PortSpec { cap: c.cap.clone(), plug: c.cap.normal[0] < 0.0 }).collect();
    let mut b = tube.bounds().expanded(3.0);
    b.min[0] -= 1.0;
    let dx = 0.25;
    let d = build_domain(&[&tube], &ports, &b, dx);
    assert_eq!(d.odd_columns, 0);
    let solid = d.cells.iter().filter(|&&c| c == SOLID).count() as f64 * dx.powi(3);
    // 48 * 30 = 1440 mm^3 of material, plus the plug in front of the inlet
    assert!((solid - 1440.0).abs() / 1440.0 < 0.08, "solid volume {solid}");
    for (k, n) in d.port_cell_counts.iter().enumerate() {
        let expect = 16.0 / (dx * dx);
        assert!((*n as f64 - expect).abs() / expect < 0.1, "port {k}: {n} vs {expect}");
    }
    assert_eq!(d.sealed_cells_removed, 0);
    let _ = dot([0.0; 3], [0.0; 3]);
}

#[test]
fn curved_cap_gets_port_cells() {
    use peregrine_core::{cap_from_loop, smooth_region_loops};
    let tube = shapes::scooped_round_tube(6.0, 4.0, 30.0, 1.5, 64).welded(1e-6);
    let seed = (0..tube.triangles.len())
        .find(|&i| {
            let [a, b, c] = tube.tri(i);
            let ctr = scale(add(add(a, b), c), 1.0 / 3.0);
            let r = (ctr[1] * ctr[1] + ctr[2] * ctr[2]).sqrt();
            ctr[0] > 27.0 && r > 4.3 && r < 5.7
        })
        .unwrap();
    let loops = smooth_region_loops(&tube, seed, 25.0);
    let inner = loops.iter().min_by(|a, b| a.perimeter.partial_cmp(&b.perimeter).unwrap()).unwrap();
    let cap = cap_from_loop(&inner.points, &tube).unwrap();
    let area = cap.area();
    let flat = find_port_candidates(&tube, 1.0).remove(0).cap;
    let ports = vec![PortSpec { cap: flat, plug: true }, PortSpec { cap: cap.clone(), plug: false }];
    let mut b = tube.bounds().expanded(3.0);
    b.max[0] += 6.0;
    let dx = 0.3;
    let d = build_domain(&[&tube], &ports, &b, dx);
    let expect = area / (dx * dx);
    let got = d.port_cell_counts[1] as f64;
    assert!((got - expect).abs() / expect < 0.25, "curved port cells {got} vs expected {expect}");
    assert_eq!(d.sealed_cells_removed, 0);
}

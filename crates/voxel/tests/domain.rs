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

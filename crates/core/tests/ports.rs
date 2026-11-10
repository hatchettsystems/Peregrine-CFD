use peregrine_core::{find_port_candidates, shapes, stl};

#[test]
fn tube_is_closed_and_outward() {
    let m = shapes::hollow_tube(8.0, 4.0, 30.0);
    let vol = m.signed_volume();
    assert!((vol - (64.0 - 16.0) * 30.0).abs() < 1e-6, "volume {vol}");
}

#[test]
fn tube_has_two_port_candidates() {
    let m = shapes::hollow_tube(8.0, 4.0, 30.0);
    let c = find_port_candidates(&m, 1.0);
    assert_eq!(c.len(), 2, "{c:?}");
    for p in &c {
        assert!((p.area - 16.0).abs() < 1e-6);
        assert!(p.cap.normal[0].abs() > 0.99);
    }
    let xs: Vec<f64> = c.iter().map(|p| p.cap.normal[0]).collect();
    assert!(xs.contains(&1.0) && xs.contains(&-1.0));
}

#[test]
fn stl_roundtrip() {
    let m = shapes::hollow_tube(8.0, 4.0, 30.0);
    let p = std::env::temp_dir().join("peregrine_tube_test.stl");
    stl::write_binary_stl(&m, &p).unwrap();
    let back = stl::read_stl(&p).unwrap();
    assert_eq!(back.triangles.len(), m.triangles.len());
    assert!((back.signed_volume() - m.signed_volume()).abs() < 1e-3);
}

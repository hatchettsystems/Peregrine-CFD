use peregrine_core::find_port_candidates;
use peregrine_occt::{read_step, write_demo_step};

#[test]
fn assembly_names_transforms_and_ports() {
    if !peregrine_occt::AVAILABLE { eprintln!("skipping: no OpenCASCADE"); return; }
    let p = std::env::temp_dir().join("peregrine_demo.step");
    write_demo_step(&p).unwrap();
    let bodies = read_step(&p, 0.05).unwrap();
    let names: Vec<&str> = bodies.iter().map(|b| b.name.as_str()).collect();
    assert!(names.contains(&"duct") && names.contains(&"nozzle"), "{names:?}");
    let duct = bodies.iter().find(|b| b.name == "duct").unwrap();
    // 10x10x40 box minus 6x6x40 channel, translated +5 mm in Z by the assembly instance
    let vol = duct.mesh.signed_volume();
    assert!((vol - 64.0 * 40.0).abs() < 1.0, "volume {vol}");
    let b = duct.mesh.bounds();
    assert!((b.min[2] - 0.0).abs() < 1e-6 && (b.max[2] - 10.0).abs() < 1e-6, "{b:?}");
    let caps = find_port_candidates(&duct.mesh, 1.0);
    assert_eq!(caps.len(), 2, "{caps:?}");
    assert!(caps.iter().all(|c| (c.area - 36.0).abs() < 1e-3));
}

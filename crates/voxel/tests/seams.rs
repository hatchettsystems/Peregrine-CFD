use peregrine_core::{find_port_candidates, shapes, Aabb};
use peregrine_voxel::{build_domain, build_domain_ext, leak_check, PortSpec, PORT_BASE};

/// Two hollow tubes joined end to end with a gap (mm) between them.
fn two_part_duct(gap: f64, dx: f64, seal_tol: f64) -> (bool, usize) {
    let mut a = shapes::hollow_tube(8.0, 4.0, 15.0);
    let mut b = shapes::hollow_tube(8.0, 4.0, 15.0);
    a.translate([0.0, 0.0, 0.0]);
    b.translate([15.0 + gap, 0.0, 0.0]);
    let mut specs = vec![];
    // inlet at x=0 (part A), outlet at the far end of part B
    for (m, inlet) in [(&a, true), (&b, false)] {
        for c in find_port_candidates(m, 1.0) {
            let want_x = if inlet { c.cap.normal[0] < 0.0 } else { c.cap.normal[0] > 0.0 };
            if want_x { specs.push(PortSpec { cap: c.cap, plug: inlet }); }
        }
    }
    assert_eq!(specs.len(), 2);
    let mut bb = Aabb::empty();
    bb.union(&a.bounds());
    bb.union(&b.bounds());
    let d = build_domain_ext(&[&a, &b], &[vec![0, 1]], seal_tol, &specs, &bb.expanded(3.0), dx);
    let fluid = d.cells.iter().filter(|&&c| c == 0).count();
    (leak_check(&d, &[PORT_BASE]).is_some(), fluid)
}

#[test]
fn seams_leak_only_when_the_gap_reaches_a_cell() {
    let dx = 0.5;
    let rows: Vec<(f64, bool)> = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0].iter().map(|&g| (g, two_part_duct(g, dx, 0.0).0)).collect();
    eprintln!("no sealing, gap -> leaks at {dx} mm cells: {rows:?}");
    assert!(!rows[0].1, "touching parts must seal");
    assert!(rows.last().unwrap().1, "a 1 mm gap must be reported as a leak");
}

#[test]
fn seam_sealing_closes_small_gaps_and_still_reports_big_ones() {
    let dx = 0.5;
    let rows: Vec<(f64, bool)> = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0].iter().map(|&g| (g, two_part_duct(g, dx, 0.5).0)).collect();
    eprintln!("sealing 0.5 mm, gap -> leaks: {rows:?}");
    for (g, leaks) in &rows {
        if *g <= 0.5 { assert!(!leaks, "a {g} mm seam should have been sealed"); }
    }
    assert!(rows.last().unwrap().1, "a 2 mm gap is too big to seal and must be reported");
    // sealing must not eat into the channel: the lumen (fluid cell count) of a sealed 0.3 mm seam is within 3% of a touching joint
    let (_, f0) = two_part_duct(0.0, dx, 0.5);
    let (_, f3) = two_part_duct(0.3, dx, 0.5);
    assert!((f0 as f64 - f3 as f64).abs() / (f0 as f64) < 0.03, "fluid cells {f0} vs {f3}");
    let _ = build_domain; // (keeps the plain entry point exercised by other tests)
}

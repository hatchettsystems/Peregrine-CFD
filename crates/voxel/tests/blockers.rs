use peregrine_core::{find_port_candidates, shapes, Aabb};
use peregrine_voxel::{build_domain_full, leak_check, PortSpec, PORT_BASE};

/// A tube with an inlet at one end and its other end open: air escapes there unless the end is blocked.
fn leaks(block_far_end: bool) -> bool {
    let tube = shapes::hollow_tube(8.0, 4.0, 20.0);
    let caps = find_port_candidates(&tube, 1.0);
    let inlet = caps.iter().find(|c| c.cap.normal[0] < 0.0).unwrap().cap.clone();
    let far = caps.iter().find(|c| c.cap.normal[0] > 0.0).unwrap().cap.clone();
    let blockers = if block_far_end { vec![far] } else { vec![] };
    let mut bb = Aabb::empty();
    bb.union(&tube.bounds());
    let d = build_domain_full(&[&tube], &[], 0.0, &[PortSpec { cap: inlet, plug: true, collar_mm: 0.0 }], &blockers, &[], &bb.expanded(3.0), 0.5);
    leak_check(&d, &[PORT_BASE]).is_some()
}

#[test]
fn a_blocker_shuts_an_opening() {
    assert!(leaks(false), "an unblocked open end must leak");
    assert!(!leaks(true), "a blocked end must not leak");
}

#[test]
fn a_plug_seals_a_leak_and_the_leak_is_located() {
    let tube = shapes::hollow_tube(8.0, 4.0, 20.0);
    let caps = find_port_candidates(&tube, 1.0);
    let inlet = caps.iter().find(|c| c.cap.normal[0] < 0.0).unwrap().cap.clone();
    let mut bb = Aabb::empty();
    bb.union(&tube.bounds());
    let specs = [PortSpec { cap: inlet, plug: true, collar_mm: 0.0 }];
    let open = build_domain_full(&[&tube], &[], 0.0, &specs, &[], &[], &bb.expanded(3.0), 0.5);
    let leak = leak_check(&open, &[PORT_BASE]).expect("the open far end leaks");
    assert!(leak.throat[0] > 5.0, "throat {:?}", leak.throat);
    // a sphere at the far end that spans the 4 mm lumen shuts it
    let sealed = build_domain_full(&[&tube], &[], 0.0, &specs, &[], &[([19.0, 0.0, 0.0], 3.0)], &bb.expanded(3.0), 0.5);
    assert!(leak_check(&sealed, &[PORT_BASE]).is_none(), "a plug must seal the leak");
}

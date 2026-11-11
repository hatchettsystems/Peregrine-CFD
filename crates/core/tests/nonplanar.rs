use peregrine_core::vec3::*;
use peregrine_core::{cap_from_loop, find_port_candidates, shapes, smooth_region_loops};

fn tube() -> peregrine_core::TriMesh {
    shapes::scooped_round_tube(6.0, 4.0, 30.0, 1.5, 64).welded(1e-6)
}

#[test]
fn tube_is_a_closed_outward_solid() {
    let m = tube();
    let expect = std::f64::consts::PI * (36.0 - 16.0) * 30.0;
    assert!((m.signed_volume() - expect).abs() / expect < 0.03, "{} vs {expect}", m.signed_volume());
}

#[test]
fn planar_detection_misses_the_scooped_end() {
    // only the flat end at x=0 is found by the planar-hole detector
    let c = find_port_candidates(&tube(), 1.0);
    assert_eq!(c.len(), 1, "{:?}", c.iter().map(|x| x.cap.centroid()).collect::<Vec<_>>());
    assert!(c[0].cap.centroid()[0].abs() < 1e-6);
}

#[test]
fn rim_click_finds_the_curved_loop_and_a_cap() {
    let m = tube();
    // a triangle on the curved rim: centroid near x ~ 30, radius between 4 and 6
    let seed = (0..m.triangles.len())
        .find(|&i| {
            let [a, b, c] = m.tri(i);
            let ctr = scale(add(add(a, b), c), 1.0 / 3.0);
            let r = (ctr[1] * ctr[1] + ctr[2] * ctr[2]).sqrt();
            ctr[0] > 27.0 && r > 4.3 && r < 5.7
        })
        .expect("rim triangle");
    let loops = smooth_region_loops(&m, seed, 25.0);
    assert_eq!(loops.len(), 2, "outer and inner rim loops");
    let inner = loops.iter().min_by(|a, b| a.perimeter.partial_cmp(&b.perimeter).unwrap()).unwrap();
    let r_mean: f64 = inner.points.iter().map(|p| (p[1] * p[1] + p[2] * p[2]).sqrt()).sum::<f64>() / inner.points.len() as f64;
    assert!((r_mean - 4.0).abs() < 0.05, "inner loop radius {r_mean}");
    assert!(inner.planarity_dev > 0.8, "loop must be clearly non-planar, dev {}", inner.planarity_dev);

    let cap = cap_from_loop(&inner.points, &m).expect("cap");
    let area_disc = std::f64::consts::PI * 16.0;
    // a saddle surface is larger than the flat disc through the same loop, but not by much
    assert!(cap.area() >= 0.98 * area_disc && cap.area() <= 1.3 * area_disc, "cap area {} vs disc {area_disc}", cap.area());
    // the duct interior is at smaller x, so the outward normal is +x
    assert!(cap.normal[0] > 0.95, "normal {:?}", cap.normal);
    assert!(cap.triangles.iter().all(|t| dot(cross(sub(t[1], t[0]), sub(t[2], t[0])), cap.normal) > 0.0), "all cap triangles face outward");
    assert!(cap.is_surface());
}

#[test]
fn automatic_detection_finds_only_the_curved_inner_opening() {
    let m = tube();
    let found = peregrine_core::find_curved_openings(&m, 10.0, 300.0, 25.0);
    assert_eq!(found.len(), 1, "{:?}", found.iter().map(|o| (o.cap.area(), o.perimeter)).collect::<Vec<_>>());
    let o = &found[0];
    let disc = std::f64::consts::PI * 16.0;
    assert!(o.cap.area() >= 0.98 * disc && o.cap.area() <= 1.3 * disc, "area {}", o.cap.area());
    assert!(o.cap.normal[0] > 0.95, "outward normal {:?}", o.cap.normal);
    assert!(o.planarity_dev > 0.8);
}

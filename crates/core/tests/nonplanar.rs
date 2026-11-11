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

#[test]
fn visibility_and_clearance_tell_real_openings_from_internal_loops() {
    use peregrine_core::{external_visibility, inward_clearance, PortCap, RayGrid};
    let tube = shapes::hollow_tube(8.0, 4.0, 20.0);
    let grid = RayGrid::new(&tube);
    // the real end opening: visible from outside, with a channel behind it
    let end = find_port_candidates(&tube, 1.0).into_iter().find(|c| c.cap.normal[0] < 0.0).unwrap().cap;
    assert!(external_visibility(&end, &grid) > 0.95, "end opening visibility {}", external_visibility(&end, &grid));
    assert!(inward_clearance(&end, &grid, 5.0) > 0.95);
    // a made-up loop inside the channel that faces the channel wall is not visible from outside
    let inner = PortCap {
        origin: [10.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        polygon: vec![[8.0, 0.0, -1.0], [12.0, 0.0, -1.0], [12.0, 0.0, 1.0], [8.0, 0.0, 1.0]],
        triangles: vec![],
        holes: vec![],
    };
    assert!(external_visibility(&inner, &grid) < 0.2, "internal loop visibility {}", external_visibility(&inner, &grid));
    // a pocket: an opening whose "inside" is a wall within a few mm (the wall faces the cap)
    let pocket = PortCap {
        origin: [10.0, 1.0, 0.0],
        normal: [0.0, -1.0, 0.0], // inward direction is +y, into the channel wall 1 mm away
        polygon: vec![[8.0, 1.0, -1.0], [12.0, 1.0, -1.0], [12.0, 1.0, 1.0], [8.0, 1.0, 1.0]],
        triangles: vec![],
        holes: vec![],
    };
    assert!(inward_clearance(&pocket, &grid, 5.0) < 0.2, "pocket clearance {}", inward_clearance(&pocket, &grid, 5.0));
}

#[test]
fn manual_loop_follows_the_rim_between_clicked_points() {
    use peregrine_core::{cap_from_loop, EdgeGraph};
    let m = tube();
    let g = EdgeGraph::new(&m, 25.0);
    // four vertices spread around the inner rim (radius 4, far end), as a user would click them
    let on_rim = |angle_deg: f64| -> u32 {
        let (s, c) = angle_deg.to_radians().sin_cos();
        let want = [0.0, 4.0 * c, 4.0 * s];
        (0..m.vertices.len() as u32)
            .filter(|&i| m.vertices[i as usize][0] > 27.0)
            .min_by(|&a, &b| {
                let da = (m.vertices[a as usize][1] - want[1]).hypot(m.vertices[a as usize][2] - want[2]) + (4.0 - (m.vertices[a as usize][1].hypot(m.vertices[a as usize][2]))).abs();
                let db = (m.vertices[b as usize][1] - want[1]).hypot(m.vertices[b as usize][2] - want[2]) + (4.0 - (m.vertices[b as usize][1].hypot(m.vertices[b as usize][2]))).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
    };
    let picks = [on_rim(0.0), on_rim(90.0), on_rim(180.0), on_rim(270.0)];
    let mut loop_pts: Vec<[f64; 3]> = vec![];
    for k in 0..4 {
        let path = g.path(picks[k], picks[(k + 1) % 4], 4.0).expect("path");
        for &v in &path[..path.len() - 1] {
            loop_pts.push(m.vertices[v as usize]);
        }
    }
    let perim: f64 = (0..loop_pts.len()).map(|i| len(sub(loop_pts[(i + 1) % loop_pts.len()], loop_pts[i]))).sum();
    // the rim is a saddle curve, so compare with the true 3D rim length found automatically
    let seed = (0..m.triangles.len()).find(|&i| { let [a, b, c] = m.tri(i); let ctr = scale(add(add(a, b), c), 1.0 / 3.0); let r = ctr[1].hypot(ctr[2]); ctr[0] > 27.0 && r > 4.3 && r < 5.7 }).unwrap();
    let auto = smooth_region_loops(&m, seed, 25.0);
    let expect = auto.iter().map(|l| l.perimeter).fold(f64::INFINITY, f64::min);
    assert!((perim - expect).abs() / expect < 0.03, "hand-traced loop length {perim} vs rim {expect}");
    // every traced point sits on the inner rim, not cutting across the wall or the rim band
    assert!(loop_pts.iter().all(|p| (p[1].hypot(p[2]) - 4.0).abs() < 0.05), "path left the inner rim");
    let cap = cap_from_loop(&loop_pts, &m).expect("cap from a hand-drawn loop");
    assert!(cap.normal[0] > 0.9);
}


#[test]
fn flipped_faces_are_turned_outward() {
    use peregrine_core::shapes;
    let mut m = shapes::solid_box([0.0; 3], [4.0, 3.0, 2.0]);
    let good_volume = m.signed_volume();
    assert!(good_volume > 0.0);
    // flip a few triangles and the whole piece for a second mesh
    for t in [0usize, 5, 7] { m.triangles[t].swap(1, 2); }
    let flipped = m.orient_outward();
    assert!(flipped >= 3, "flipped {flipped}");
    assert!((m.signed_volume() - good_volume).abs() < 1e-9);
    let mut inside_out = shapes::solid_box([0.0; 3], [4.0, 3.0, 2.0]);
    for t in inside_out.triangles.iter_mut() { t.swap(1, 2); }
    inside_out.orient_outward();
    assert!(inside_out.signed_volume() > 0.0);
}

use peregrine_core::{find_curved_openings, stl};
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let m = stl::read_stl(std::path::Path::new(&p)).unwrap();
    let t = std::time::Instant::now();
    let c = find_curved_openings(&m, 12.0, 400.0, 25.0);
    println!("{} curved openings in {:.1}s", c.len(), t.elapsed().as_secs_f64());
    for (i, o) in c.iter().enumerate().take(25) {
        let ctr = o.cap.centroid();
        println!("#{i}: area {:>7.1} mm2  perimeter {:>6.1}  off-plane {:>4.1} mm  conf {:.2}  centre ({:.1},{:.1},{:.1}) normal ({:.2},{:.2},{:.2})", o.cap.area(), o.perimeter, o.planarity_dev, o.confidence, ctr[0], ctr[1], ctr[2], o.cap.normal[0], o.cap.normal[1], o.cap.normal[2]);
    }
}

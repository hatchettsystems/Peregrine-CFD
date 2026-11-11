use peregrine_core::{shapes, stl};
fn main() {
    let out = std::env::args().nth(1).unwrap();
    let m = shapes::scooped_round_tube(8.0, 5.5, 40.0, 3.0, 96);
    stl::write_binary_stl(&m, std::path::Path::new(&out)).unwrap();
}

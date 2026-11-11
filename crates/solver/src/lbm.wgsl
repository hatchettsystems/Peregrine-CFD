// D3Q19 two-relaxation-time LBM with Smagorinsky eddy viscosity. Mirrors src/lbm.rs.
struct Params {
    dims: vec4<u32>,        // nx, ny, nz, n_cells
    tau0_cs_nport: vec4<f32>, // tau0, smagorinsky, (unused), (unused)
    force: vec4<f32>,
    flags: vec4<u32>,       // open-face bitmask, n_port_cells, 0, 0
};
struct PortGpu {
    vel: vec4<f32>,         // lattice velocity (kind 0)
    prm: vec4<f32>,         // x: kind (0 velocity, 1 pressure), y: rho (kind 1)
};
struct PortCell { idx: u32, k: u32, nb: u32, pad: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@group(0) @binding(2) var<storage, read> ports: array<PortGpu>;
@group(0) @binding(3) var<storage, read> port_cells: array<PortCell>;
@group(0) @binding(4) var<storage, read_write> mac: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> f_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> f_out: array<f32>;

const Q: u32 = 19u;
var<private> CX: array<i32, 19> = array<i32, 19>(0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0);
var<private> CY: array<i32, 19> = array<i32, 19>(0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1);
var<private> CZ: array<i32, 19> = array<i32, 19>(0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1);

fn weight(i: u32) -> f32 {
    if (i == 0u) { return 1.0 / 3.0; }
    if (i <= 6u) { return 1.0 / 18.0; }
    return 1.0 / 36.0;
}
fn opp(i: u32) -> u32 {
    if (i == 0u) { return 0u; }
    if ((i & 1u) == 1u) { return i + 1u; }
    return i - 1u;
}
fn feq(i: u32, rho: f32, u: vec3<f32>) -> f32 {
    let cu = f32(CX[i]) * u.x + f32(CY[i]) * u.y + f32(CZ[i]) * u.z;
    let uu = dot(u, u);
    return weight(i) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
}

@compute @workgroup_size(64)
fn collide_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 64u;
    let n = P.dims.w;
    if (idx >= n) { return; }
    if (cells[idx] != 0u) { return; }
    let nx = P.dims.x; let ny = P.dims.y; let nz = P.dims.z;
    let x = i32(idx % nx);
    let y = i32((idx / nx) % ny);
    let z = i32(idx / (nx * ny));
    var fi: array<f32, 19>;
    for (var i = 0u; i < Q; i = i + 1u) {
        let sx = x - CX[i]; let sy = y - CY[i]; let sz = z - CZ[i];
        var face = -1;
        if (sx < 0) { face = 0; }
        else if (sx >= i32(nx)) { face = 1; }
        else if (sy < 0) { face = 2; }
        else if (sy >= i32(ny)) { face = 3; }
        else if (sz < 0) { face = 4; }
        else if (sz >= i32(nz)) { face = 5; }
        if (face >= 0) {
            var val = f_in[opp(i) * n + idx];
            if (((P.flags.x >> u32(face)) & 1u) == 1u) {
                var ix = x; var iy = y; var iz = z;
                let ax = face / 2;
                let step = select(-1, 1, (face % 2) == 0);
                if (ax == 0) { ix = ix + step; }
                else if (ax == 1) { iy = iy + step; }
                else { iz = iz + step; }
                if (ix >= 0 && ix < i32(nx) && iy >= 0 && iy < i32(ny) && iz >= 0 && iz < i32(nz)) {
                    let j = u32(ix) + nx * (u32(iy) + ny * u32(iz));
                    if (cells[j] == 0u) {
                        // pressure boundary: rho = 1 with the neighbour's velocity plus its non-equilibrium part
                        var rj = 0.0;
                        var mj = vec3<f32>(0.0);
                        for (var k = 0u; k < Q; k = k + 1u) {
                            let fk = f_in[k * n + j];
                            rj = rj + fk;
                            mj = mj + vec3<f32>(f32(CX[k]), f32(CY[k]), f32(CZ[k])) * fk;
                        }
                        let uj = mj / rj;
                        val = feq(i, 1.0, uj) + (f_in[i * n + j] - feq(i, rj, uj));
                    }
                }
            }
            fi[i] = val;
        } else {
            let j = u32(sx) + nx * (u32(sy) + ny * u32(sz));
            if (cells[j] == 1u) { fi[i] = f_in[opp(i) * n + idx]; }
            else { fi[i] = f_in[i * n + j]; }
        }
    }
    var rho = 0.0;
    var m = vec3<f32>(0.0);
    for (var i = 0u; i < Q; i = i + 1u) {
        rho = rho + fi[i];
        m = m + vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i])) * fi[i];
    }
    let inv = 1.0 / rho;
    let u = m * inv;
    let tau0 = P.tau0_cs_nport.x;
    let cs = P.tau0_cs_nport.y;
    var tau = tau0;
    let reg = P.flags.z == 1u;
    var pxx = 0.0; var pyy = 0.0; var pzz = 0.0; var pxy = 0.0; var pxz = 0.0; var pyz = 0.0;
    if (cs > 0.0 || reg) {
        for (var i = 0u; i < Q; i = i + 1u) {
            let fn_ = fi[i] - feq(i, rho, u);
            let cx = f32(CX[i]); let cy = f32(CY[i]); let cz = f32(CZ[i]);
            pxx = pxx + cx * cx * fn_; pyy = pyy + cy * cy * fn_; pzz = pzz + cz * cz * fn_;
            pxy = pxy + cx * cy * fn_; pxz = pxz + cx * cz * fn_; pyz = pyz + cy * cz * fn_;
        }
    }
    if (cs > 0.0) {
        let s = pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz);
        tau = 0.5 * (tau0 + sqrt(tau0 * tau0 + 18.0 * 1.41421356 * cs * cs * sqrt(s) * inv));
    }
    let sponge = i32(P.flags.w);
    if (sponge > 0) {
        var dmin = 1000000;
        let dist = array<i32, 6>(x, i32(nx) - 1 - x, y, i32(ny) - 1 - y, z, i32(nz) - 1 - z);
        for (var f = 0u; f < 6u; f = f + 1u) {
            if (((P.flags.x >> f) & 1u) == 1u) { dmin = min(dmin, dist[f]); }
        }
        if (dmin < sponge) {
            let w = f32(sponge - dmin) / f32(sponge);
            tau = tau + w * w * (1.0 - tau);
        }
    }
    let omega = 1.0 / tau;
    let tau_m = 0.5 + 0.1875 / (tau - 0.5);
    let omega_m = 1.0 / tau_m;
    let force = P.force.xyz;
    let ueq = u + tau * force * inv;
    let uphys = u + 0.5 * force * inv;
    var fe: array<f32, 19>;
    for (var i = 0u; i < Q; i = i + 1u) { fe[i] = feq(i, rho, ueq); }
    if (reg) {
        for (var i = 0u; i < Q; i = i + 1u) {
            let cx = f32(CX[i]); let cy = f32(CY[i]); let cz = f32(CZ[i]);
            let q = (cx * cx - 1.0 / 3.0) * pxx + (cy * cy - 1.0 / 3.0) * pyy + (cz * cz - 1.0 / 3.0) * pzz
                + 2.0 * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz);
            f_out[i * n + idx] = fe[i] + (1.0 - omega) * weight(i) * 4.5 * q;
        }
    } else {
        f_out[idx] = fi[0] - omega * (fi[0] - fe[0]);
        for (var i = 1u; i < Q; i = i + 1u) {
            let o = opp(i);
            let fp = 0.5 * (fi[i] + fi[o]);
            let fm = 0.5 * (fi[i] - fi[o]);
            let ep = 0.5 * (fe[i] + fe[o]);
            let em = 0.5 * (fe[i] - fe[o]);
            f_out[i * n + idx] = fi[i] - omega * (fp - ep) - omega_m * (fm - em);
        }
    }
    mac[idx] = vec4<f32>(rho, uphys);
}

@compute @workgroup_size(64)
fn port_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= P.flags.y) { return; }
    let pc = port_cells[t];
    let n = P.dims.w;
    let pg = ports[pc.k];
    let mnb = mac[pc.nb];
    var rho = mnb.x;
    var u = pg.vel.xyz;
    if (pg.prm.x > 0.5) {
        rho = pg.prm.y;
        u = mnb.yzw;
    }
    for (var i = 0u; i < Q; i = i + 1u) {
        var neq = 0.0;
        if (pc.nb != pc.idx) { neq = f_out[i * n + pc.nb] - feq(i, mnb.x, mnb.yzw); }
        f_out[i * n + pc.idx] = feq(i, rho, u) + neq;
    }
    mac[pc.idx] = vec4<f32>(rho, u);
}

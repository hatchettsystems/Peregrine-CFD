//! wgpu compute backend (D3Q19 TRT + Smagorinsky). Runs on Vulkan / DX12 / Metal.
use crate::lbm::{port_cell_list, Engine, LbmConfig, PortBc};
use anyhow::{anyhow, bail, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    dims: [u32; 4],
    tau0_cs: [f32; 4],
    force: [f32; 4],
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PortGpu {
    vel: [f32; 4],
    prm: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PortCellGpu {
    idx: u32,
    k: u32,
    nb: u32,
    pad: u32,
}

pub struct GpuLbm {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    collide: wgpu::ComputePipeline,
    port_update: wgpu::ComputePipeline,
    bind: [wgpu::BindGroup; 2],
    ports_buf: wgpu::Buffer,
    mac_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    ports: Vec<PortGpu>,
    n: usize,
    n_port_cells: u32,
    flip: usize,
    steps: usize,
    mac_cache: Vec<[f32; 4]>,
    cache_valid: bool,
    groups: (u32, u32),
    port_groups: u32,
}

fn port_to_gpu(bc: PortBc) -> PortGpu {
    match bc {
        PortBc::Velocity(u) => PortGpu { vel: [u[0], u[1], u[2], 0.0], prm: [0.0; 4] },
        PortBc::Pressure(r) => PortGpu { vel: [0.0; 4], prm: [1.0, r, 0.0, 0.0] },
    }
}

impl GpuLbm {
    pub fn new(cfg: &LbmConfig) -> Result<GpuLbm> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok_or_else(|| anyhow!("no GPU adapter found"))?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let n = cfg.cells.len();
        let f_bytes = (n * 19 * 4) as u64;
        if f_bytes > limits.max_storage_buffer_binding_size as u64 || f_bytes > limits.max_buffer_size {
            bail!("grid of {} cells needs a {} MB buffer; this GPU allows {} MB per buffer", n, f_bytes >> 20, (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size) >> 20);
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("peregrine"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lbm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lbm.wgsl").into()),
        });

        // initial equilibrium at rest, SoA layout f[i*n + idx]
        let w = |i: usize| crate::lbm::W[i];
        let mut f0 = vec![0f32; n * 19];
        for i in 0..19 {
            for idx in 0..n {
                if cfg.cells[idx] != 1 {
                    f0[i * n + idx] = w(i);
                }
            }
        }
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let fa = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("fa"), contents: bytemuck::cast_slice(&f0), usage: storage });
        let fb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("fb"), contents: bytemuck::cast_slice(&f0), usage: storage });
        drop(f0);
        let cells_u32: Vec<u32> = cfg.cells.iter().map(|&c| c as u32).collect();
        let cells_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("cells"), contents: bytemuck::cast_slice(&cells_u32), usage: wgpu::BufferUsages::STORAGE });
        let mac0 = vec![[1.0f32, 0.0, 0.0, 0.0]; n];
        let mac_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("mac"), contents: bytemuck::cast_slice(&mac0), usage: storage });
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("staging"), size: (n * 16) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let ports: Vec<PortGpu> = cfg.ports.iter().map(|p| port_to_gpu(p.bc)).collect();
        let ports_init: Vec<PortGpu> = if ports.is_empty() { vec![PortGpu::zeroed()] } else { ports.clone() };
        let ports_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ports"), contents: bytemuck::cast_slice(&ports_init), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST });
        let pcs: Vec<PortCellGpu> = port_cell_list(cfg).iter().map(|&(idx, k, nb)| PortCellGpu { idx: idx as u32, k: k as u32, nb: nb as u32, pad: 0 }).collect();
        let n_port_cells = pcs.len() as u32;
        let pcs_init = if pcs.is_empty() { vec![PortCellGpu::zeroed()] } else { pcs };
        let pc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("portcells"), contents: bytemuck::cast_slice(&pcs_init), usage: wgpu::BufferUsages::STORAGE });

        let mut open_mask = 0u32;
        for (i, o) in cfg.open_faces.iter().enumerate() {
            if *o {
                open_mask |= 1 << i;
            }
        }
        let params = Params {
            dims: [cfg.dims[0] as u32, cfg.dims[1] as u32, cfg.dims[2] as u32, n as u32],
            tau0_cs: [cfg.tau0, cfg.smagorinsky, 0.0, 0.0],
            force: [cfg.body_force[0], cfg.body_force[1], cfg.body_force[2], 0.0],
            flags: [open_mask, n_port_cells, cfg.regularized as u32, cfg.sponge_cells as u32],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });

        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lbm layout"),
            entries: &[entry(0, wgpu::BufferBindingType::Uniform), entry(1, ro), entry(2, ro), entry(3, ro), entry(4, rw), entry(5, ro), entry(6, rw)],
        });
        let make_bind = |fin: &wgpu::Buffer, fout: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ports_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: pc_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: mac_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: fin.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: fout.as_entire_binding() },
                ],
            })
        };
        let bind = [make_bind(&fa, &fb), make_bind(&fb, &fa)];
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&layout], push_constant_ranges: &[] });
        let mk = |name: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some(name),
            compilation_options: Default::default(),
            cache: None,
        });
        let collide = mk("collide_stream");
        let port_update = mk("port_update");
        // dispatch in 2D so very large grids exceed neither the 65535 group limit
        let total_groups = (n as u32 + 63) / 64;
        let gx = total_groups.min(65535);
        let gy = (total_groups + gx - 1) / gx;
        Ok(GpuLbm {
            device, queue, adapter_name: format!("{} ({:?})", info.name, info.backend), collide, port_update, bind, ports_buf, mac_buf, staging,
            ports, n, n_port_cells, flip: 0, steps: 0, mac_cache: vec![[1.0, 0.0, 0.0, 0.0]; n], cache_valid: true,
            groups: (gx, gy), port_groups: (n_port_cells + 63) / 64,
        })
    }
}

impl Engine for GpuLbm {
    fn step_n(&mut self, count: usize) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..count {
                pass.set_bind_group(0, &self.bind[self.flip], &[]);
                pass.set_pipeline(&self.collide);
                pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
                if self.n_port_cells > 0 {
                    pass.set_pipeline(&self.port_update);
                    pass.dispatch_workgroups(self.port_groups, 1, 1);
                }
                self.flip ^= 1;
            }
        }
        self.queue.submit(Some(enc.finish()));
        self.steps += count;
        self.cache_valid = false;
    }

    fn set_port_bc(&mut self, k: usize, bc: PortBc) {
        self.ports[k] = port_to_gpu(bc);
        self.queue.write_buffer(&self.ports_buf, 0, bytemuck::cast_slice(&self.ports));
    }

    fn mac(&mut self) -> &[[f32; 4]] {
        if !self.cache_valid {
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&self.mac_buf, 0, &self.staging, 0, (self.n * 16) as u64);
            self.queue.submit(Some(enc.finish()));
            let slice = self.staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().expect("map failed");
            {
                let data = slice.get_mapped_range();
                self.mac_cache.copy_from_slice(bytemuck::cast_slice(&data));
            }
            self.staging.unmap();
            self.cache_valid = true;
        }
        &self.mac_cache
    }

    fn steps(&self) -> usize { self.steps }
    fn name(&self) -> String { format!("GPU {}", self.adapter_name) }
}

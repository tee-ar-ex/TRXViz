use crate::data::orientation_field::BoundaryContactField;
use glam::Vec3;
use lin_alg::f32::Vec3 as LinVec3;
use mcubes::{MarchingCubes, MeshSide};

/// Per-vertex data for a bundle surface mesh.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BundleMeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Clone)]
pub struct BundleMesh {
    pub vertices: Vec<BundleMeshVertex>,
    pub indices: Vec<u32>,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum BundleMeshColorStrategy {
    SampledRgb,
    DominantOrientation,
    BoundaryField,
    Constant([f32; 4]),
}

// ── Voxel color grid ─────────────────────────────────────────────────────────

struct ColorGrid {
    density: Vec<f32>,
    r_sum: Vec<f32>,
    g_sum: Vec<f32>,
    b_sum: Vec<f32>,
    xx_sum: Vec<f32>,
    xy_sum: Vec<f32>,
    xz_sum: Vec<f32>,
    yy_sum: Vec<f32>,
    yz_sum: Vec<f32>,
    zz_sum: Vec<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl ColorGrid {
    /// mcubes flat index: x is fastest-changing, z is slowest.
    fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + iy * self.nx + iz * self.nx * self.ny
    }

    fn voxel_color(&self, ix: usize, iy: usize, iz: usize) -> [f32; 3] {
        let i = self.idx(ix, iy, iz);
        let d = self.density[i];
        if d > 0.0 {
            [self.r_sum[i] / d, self.g_sum[i] / d, self.b_sum[i] / d]
        } else {
            [0.5, 0.5, 0.5]
        }
    }

    fn voxel_tensor(&self, ix: usize, iy: usize, iz: usize) -> [f32; 6] {
        let i = self.idx(ix, iy, iz);
        let d = self.density[i];
        if d > 0.0 {
            [
                self.xx_sum[i] / d,
                self.xy_sum[i] / d,
                self.xz_sum[i] / d,
                self.yy_sum[i] / d,
                self.yz_sum[i] / d,
                self.zz_sum[i] / d,
            ]
        } else {
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }

    /// Trilinear color interpolation at an arbitrary grid-space position.
    fn sample_color(&self, gx: f32, gy: f32, gz: f32) -> [f32; 4] {
        let x0 = (gx as usize).min(self.nx.saturating_sub(1));
        let y0 = (gy as usize).min(self.ny.saturating_sub(1));
        let z0 = (gz as usize).min(self.nz.saturating_sub(1));
        let x1 = (x0 + 1).min(self.nx.saturating_sub(1));
        let y1 = (y0 + 1).min(self.ny.saturating_sub(1));
        let z1 = (z0 + 1).min(self.nz.saturating_sub(1));
        let fx = gx.fract();
        let fy = gy.fract();
        let fz = gz.fract();

        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };

        let c = lerp(
            lerp(
                lerp(
                    self.voxel_color(x0, y0, z0),
                    self.voxel_color(x1, y0, z0),
                    fx,
                ),
                lerp(
                    self.voxel_color(x0, y1, z0),
                    self.voxel_color(x1, y1, z0),
                    fx,
                ),
                fy,
            ),
            lerp(
                lerp(
                    self.voxel_color(x0, y0, z1),
                    self.voxel_color(x1, y0, z1),
                    fx,
                ),
                lerp(
                    self.voxel_color(x0, y1, z1),
                    self.voxel_color(x1, y1, z1),
                    fx,
                ),
                fy,
            ),
            fz,
        );
        [c[0], c[1], c[2], 1.0]
    }

    fn sample_tensor(&self, gx: f32, gy: f32, gz: f32) -> [f32; 6] {
        let x0 = (gx as usize).min(self.nx.saturating_sub(1));
        let y0 = (gy as usize).min(self.ny.saturating_sub(1));
        let z0 = (gz as usize).min(self.nz.saturating_sub(1));
        let x1 = (x0 + 1).min(self.nx.saturating_sub(1));
        let y1 = (y0 + 1).min(self.ny.saturating_sub(1));
        let z1 = (z0 + 1).min(self.nz.saturating_sub(1));
        let fx = gx.fract();
        let fy = gy.fract();
        let fz = gz.fract();

        let lerp = |a: [f32; 6], b: [f32; 6], t: f32| -> [f32; 6] {
            let mut out = [0.0; 6];
            for i in 0..6 {
                out[i] = a[i] + (b[i] - a[i]) * t;
            }
            out
        };

        lerp(
            lerp(
                lerp(
                    self.voxel_tensor(x0, y0, z0),
                    self.voxel_tensor(x1, y0, z0),
                    fx,
                ),
                lerp(
                    self.voxel_tensor(x0, y1, z0),
                    self.voxel_tensor(x1, y1, z0),
                    fx,
                ),
                fy,
            ),
            lerp(
                lerp(
                    self.voxel_tensor(x0, y0, z1),
                    self.voxel_tensor(x1, y0, z1),
                    fx,
                ),
                lerp(
                    self.voxel_tensor(x0, y1, z1),
                    self.voxel_tensor(x1, y1, z1),
                    fx,
                ),
                fy,
            ),
            fz,
        )
    }
}

fn principal_direction_rgb(tensor: [f32; 6]) -> [f32; 4] {
    let [xx, xy, xz, yy, yz, zz] = tensor;
    let mul = |v: Vec3| -> Vec3 {
        Vec3::new(
            xx * v.x + xy * v.y + xz * v.z,
            xy * v.x + yy * v.y + yz * v.z,
            xz * v.x + yz * v.y + zz * v.z,
        )
    };

    let mut v = if xx >= yy && xx >= zz {
        Vec3::X
    } else if yy >= zz {
        Vec3::Y
    } else {
        Vec3::Z
    };

    for _ in 0..6 {
        let next = mul(v);
        if next.length_squared() < 1e-8 {
            break;
        }
        v = next.normalize();
    }

    [v.x.abs(), v.y.abs(), v.z.abs(), 1.0]
}

fn color_strategy_for_point(
    strategy: BundleMeshColorStrategy,
    grid: &ColorGrid,
    boundary_field: Option<&BoundaryContactField>,
    world: Vec3,
    gx: f32,
    gy: f32,
    gz: f32,
) -> [f32; 4] {
    match strategy {
        BundleMeshColorStrategy::SampledRgb => grid.sample_color(gx, gy, gz),
        BundleMeshColorStrategy::DominantOrientation => {
            principal_direction_rgb(grid.sample_tensor(gx, gy, gz))
        }
        BundleMeshColorStrategy::BoundaryField => {
            if let Some(field) = boundary_field {
                let v = field.sample_summary_vector(world);
                if v.length_squared() > 1e-8 {
                    let rgb = v.normalize().abs();
                    [rgb.x, rgb.y, rgb.z, 1.0]
                } else {
                    [0.7, 0.7, 0.7, 1.0]
                }
            } else {
                [0.7, 0.7, 0.7, 1.0]
            }
        }
        BundleMeshColorStrategy::Constant(color) => color,
    }
}

// ── Gaussian blur (separable) ─────────────────────────────────────────────────

/// 3-D separable Gaussian blur applied to a flat voxel grid.
/// Returns the input unchanged when `sigma < 0.5`.
fn gaussian_blur_3d(data: &[f32], nx: usize, ny: usize, nz: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.5 {
        return data.to_vec();
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel: Vec<f32> = (0..size)
        .map(|i| {
            let x = i as f32 - radius as f32;
            (-0.5 * x * x / (sigma * sigma)).exp()
        })
        .collect();
    let sum: f32 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= sum;
    }

    let mut src = data.to_vec();
    let mut dst = vec![0.0f32; data.len()];

    // X pass
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut val = 0.0f32;
                for (ki, &k) in kernel.iter().enumerate() {
                    let sx = (ix as isize + ki as isize - radius as isize).clamp(0, nx as isize - 1)
                        as usize;
                    val += k * src[sx + iy * nx + iz * nx * ny];
                }
                dst[ix + iy * nx + iz * nx * ny] = val;
            }
        }
    }
    std::mem::swap(&mut src, &mut dst);

    // Y pass
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut val = 0.0f32;
                for (ki, &k) in kernel.iter().enumerate() {
                    let sy = (iy as isize + ki as isize - radius as isize).clamp(0, ny as isize - 1)
                        as usize;
                    val += k * src[ix + sy * nx + iz * nx * ny];
                }
                dst[ix + iy * nx + iz * nx * ny] = val;
            }
        }
    }
    std::mem::swap(&mut src, &mut dst);

    // Z pass
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut val = 0.0f32;
                for (ki, &k) in kernel.iter().enumerate() {
                    let sz = (iz as isize + ki as isize - radius as isize).clamp(0, nz as isize - 1)
                        as usize;
                    val += k * src[ix + iy * nx + sz * nx * ny];
                }
                dst[ix + iy * nx + iz * nx * ny] = val;
            }
        }
    }

    dst
}

// ── Connected components ─────────────────────────────────────────────────────

#[derive(Clone)]
struct TriangleComponent {
    indices: Vec<u32>,
    volume_mm3: f32,
}

/// Partition triangles into connected components.
///
/// Connectivity is determined by shared vertex positions rather than shared
/// indices, because `mcubes` may emit duplicate vertices for adjacent triangles.
fn connected_components(vertices: &[BundleMeshVertex], indices: &[u32]) -> Vec<TriangleComponent> {
    let n_tris = indices.len() / 3;
    if n_tris == 0 {
        return Vec::new();
    }
    if n_tris == 1 {
        return vec![TriangleComponent {
            indices: indices.to_vec(),
            volume_mm3: component_volume_mm3(vertices, indices),
        }];
    }

    // Build quantized-position → triangle list.
    // 0.1 mm quantization handles floating-point near-duplicates.
    const QUANT: f32 = 1e-4;
    let mut pos_tris: std::collections::HashMap<(i32, i32, i32), Vec<u32>> =
        std::collections::HashMap::new();
    for (ti, tri) in indices.chunks(3).enumerate() {
        for &vi in tri {
            let p = vertices[vi as usize].position;
            let key = (
                (p[0] / QUANT).round() as i32,
                (p[1] / QUANT).round() as i32,
                (p[2] / QUANT).round() as i32,
            );
            pos_tris.entry(key).or_default().push(ti as u32);
        }
    }

    let mut component: Vec<u32> = vec![u32::MAX; n_tris];
    let mut components = Vec::<TriangleComponent>::new();
    let mut queue: Vec<usize> = Vec::new();

    for start in 0..n_tris {
        if component[start] != u32::MAX {
            continue;
        }
        let comp_id = components.len() as u32;
        queue.clear();
        queue.push(start);
        component[start] = comp_id;
        let mut head = 0;
        while head < queue.len() {
            let ti = queue[head];
            head += 1;
            for &vi in &indices[ti * 3..ti * 3 + 3] {
                let p = vertices[vi as usize].position;
                let key = (
                    (p[0] / QUANT).round() as i32,
                    (p[1] / QUANT).round() as i32,
                    (p[2] / QUANT).round() as i32,
                );
                if let Some(neighbors) = pos_tris.get(&key) {
                    for &nti in neighbors {
                        let nti = nti as usize;
                        if component[nti] == u32::MAX {
                            component[nti] = comp_id;
                            queue.push(nti);
                        }
                    }
                }
            }
        }
        let component_indices = indices
            .chunks(3)
            .enumerate()
            .filter(|(ti, _)| component[*ti] == comp_id)
            .flat_map(|(_, tri)| tri.iter().copied())
            .collect::<Vec<_>>();
        let volume_mm3 = component_volume_mm3(vertices, &component_indices);
        components.push(TriangleComponent {
            indices: component_indices,
            volume_mm3,
        });
    }

    components
}

fn component_volume_mm3(vertices: &[BundleMeshVertex], indices: &[u32]) -> f32 {
    let mut signed_volume = 0.0f32;
    for tri in indices.chunks_exact(3) {
        let a = Vec3::from(vertices[tri[0] as usize].position);
        let b = Vec3::from(vertices[tri[1] as usize].position);
        let c = Vec3::from(vertices[tri[2] as usize].position);
        signed_volume += a.dot(b.cross(c)) / 6.0;
    }
    signed_volume.abs()
}

const WELD_QUANT: f32 = 1e-4;
const TAUBIN_SMOOTHING_ITERS: usize = 4;
const TAUBIN_LAMBDA: f32 = 0.33;
const TAUBIN_MU: f32 = -0.34;

fn quantized_position_key(p: [f32; 3]) -> (i32, i32, i32) {
    (
        (p[0] / WELD_QUANT).round() as i32,
        (p[1] / WELD_QUANT).round() as i32,
        (p[2] / WELD_QUANT).round() as i32,
    )
}

fn welded_vertex_groups(vertices: &[BundleMeshVertex]) -> (Vec<usize>, Vec<Vec3>, Vec<u32>) {
    let mut group_lookup = std::collections::HashMap::<(i32, i32, i32), usize>::new();
    let mut vertex_group = vec![0usize; vertices.len()];
    let mut group_positions = Vec::<Vec3>::new();
    let mut group_counts = Vec::<u32>::new();

    for (vi, vertex) in vertices.iter().enumerate() {
        let key = quantized_position_key(vertex.position);
        let gid = if let Some(&gid) = group_lookup.get(&key) {
            gid
        } else {
            let gid = group_positions.len();
            group_lookup.insert(key, gid);
            group_positions.push(Vec3::ZERO);
            group_counts.push(0);
            gid
        };
        vertex_group[vi] = gid;
        group_positions[gid] += Vec3::from(vertex.position);
        group_counts[gid] += 1;
    }

    for (gid, pos) in group_positions.iter_mut().enumerate() {
        *pos /= group_counts[gid].max(1) as f32;
    }

    (vertex_group, group_positions, group_counts)
}

fn group_neighbors(vertex_group: &[usize], indices: &[u32], group_count: usize) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::<usize>::new(); group_count];

    let mut connect = |a: usize, b: usize| {
        if a != b && !neighbors[a].contains(&b) {
            neighbors[a].push(b);
        }
    };

    for tri in indices.chunks_exact(3) {
        let ga = vertex_group[tri[0] as usize];
        let gb = vertex_group[tri[1] as usize];
        let gc = vertex_group[tri[2] as usize];
        connect(ga, gb);
        connect(gb, ga);
        connect(gb, gc);
        connect(gc, gb);
        connect(gc, ga);
        connect(ga, gc);
    }

    neighbors
}

fn apply_taubin_smoothing(group_positions: &mut [Vec3], neighbors: &[Vec<usize>]) {
    fn smooth_step(group_positions: &mut [Vec3], neighbors: &[Vec<usize>], factor: f32) {
        let previous = group_positions.to_vec();
        for (gid, position) in group_positions.iter_mut().enumerate() {
            let adjacent = &neighbors[gid];
            if adjacent.is_empty() {
                continue;
            }
            let average = adjacent
                .iter()
                .fold(Vec3::ZERO, |acc, &neighbor| acc + previous[neighbor])
                / adjacent.len() as f32;
            *position = previous[gid] + factor * (average - previous[gid]);
        }
    }

    for _ in 0..TAUBIN_SMOOTHING_ITERS {
        smooth_step(group_positions, neighbors, TAUBIN_LAMBDA);
        smooth_step(group_positions, neighbors, TAUBIN_MU);
    }
}

fn weld_and_recompute_normals(vertices: &mut [BundleMeshVertex], indices: &[u32]) {
    if vertices.is_empty() || indices.len() < 3 {
        return;
    }

    let (vertex_group, mut group_positions, _) = welded_vertex_groups(vertices);
    let neighbors = group_neighbors(&vertex_group, indices, group_positions.len());
    apply_taubin_smoothing(&mut group_positions, &neighbors);

    for (vi, vertex) in vertices.iter_mut().enumerate() {
        vertex.position = group_positions[vertex_group[vi]].to_array();
    }

    let mut group_normals = vec![Vec3::ZERO; group_positions.len()];
    for tri in indices.chunks_exact(3) {
        let ia = tri[0] as usize;
        let ib = tri[1] as usize;
        let ic = tri[2] as usize;
        let a = Vec3::from(vertices[ia].position);
        let b = Vec3::from(vertices[ib].position);
        let c = Vec3::from(vertices[ic].position);
        let n = (b - a).cross(c - a).normalize_or_zero();
        if n.length_squared() <= 1e-10 {
            continue;
        }
        group_normals[vertex_group[ia]] += n;
        group_normals[vertex_group[ib]] += n;
        group_normals[vertex_group[ic]] += n;
    }

    for (vi, vertex) in vertices.iter_mut().enumerate() {
        let n = group_normals[vertex_group[vi]].normalize_or_zero();
        vertex.normal = if n.length_squared() > 0.0 {
            n.to_array()
        } else {
            [0.0, 0.0, 1.0]
        };
    }
}

// ── Public entry point ───────────────────────────────────────────────────────

/// Build a surface mesh from a set of 3-D point positions and per-point colors.
///
/// * `voxel_size`   — spatial resolution in mm (smaller = tighter / more detail)
/// * `threshold`    — density (point count per voxel) at which the surface is placed
/// * `smooth_sigma` — Gaussian blur sigma in voxels applied to the density field
///                    before marching cubes (0 = off; 1-2 recommended)
pub fn build_bundle_mesh(
    positions: &[[f32; 3]],
    colors: &[[f32; 4]],
    voxel_size: f32,
    threshold: f32,
    smooth_sigma: f32,
    min_component_volume_mm3: f32,
    color_strategy: BundleMeshColorStrategy,
    boundary_field: Option<&BoundaryContactField>,
) -> Option<BundleMesh> {
    if positions.is_empty() {
        return None;
    }

    let vs = voxel_size.max(0.5);

    // ── 1. Bounding box + 2-voxel padding ───────────────────────────────────
    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    for p in positions {
        for i in 0..3 {
            mn[i] = mn[i].min(p[i]);
            mx[i] = mx[i].max(p[i]);
        }
    }
    let pad = vs * 2.0;
    for i in 0..3 {
        mn[i] -= pad;
        mx[i] += pad;
    }

    // ── 2. Grid dimensions ──────────────────────────────────────────────────
    let nx = (((mx[0] - mn[0]) / vs).ceil() as usize + 1).max(3);
    let ny = (((mx[1] - mn[1]) / vs).ceil() as usize + 1).max(3);
    let nz = (((mx[2] - mn[2]) / vs).ceil() as usize + 1).max(3);
    let n = nx * ny * nz;

    let mut grid = ColorGrid {
        density: vec![0.0f32; n],
        r_sum: vec![0.0f32; n],
        g_sum: vec![0.0f32; n],
        b_sum: vec![0.0f32; n],
        xx_sum: vec![0.0f32; n],
        xy_sum: vec![0.0f32; n],
        xz_sum: vec![0.0f32; n],
        yy_sum: vec![0.0f32; n],
        yz_sum: vec![0.0f32; n],
        zz_sum: vec![0.0f32; n],
        nx,
        ny,
        nz,
    };

    // ── 3. Voxelise ─────────────────────────────────────────────────────────
    for (pos, col) in positions.iter().zip(colors.iter()) {
        let ix = ((pos[0] - mn[0]) / vs) as usize;
        let iy = ((pos[1] - mn[1]) / vs) as usize;
        let iz = ((pos[2] - mn[2]) / vs) as usize;
        if ix >= nx || iy >= ny || iz >= nz {
            continue;
        }
        let i = grid.idx(ix, iy, iz);
        grid.density[i] += 1.0;
        grid.r_sum[i] += col[0];
        grid.g_sum[i] += col[1];
        grid.b_sum[i] += col[2];
        let dir = Vec3::new(col[0], col[1], col[2]).normalize_or_zero();
        grid.xx_sum[i] += dir.x * dir.x;
        grid.xy_sum[i] += dir.x * dir.y;
        grid.xz_sum[i] += dir.x * dir.z;
        grid.yy_sum[i] += dir.y * dir.y;
        grid.yz_sum[i] += dir.y * dir.z;
        grid.zz_sum[i] += dir.z * dir.z;
    }

    // ── 4. Gaussian blur of density field ───────────────────────────────────
    // Color grid is kept unblurred so per-vertex colors stay accurate.
    let blurred_density = gaussian_blur_3d(&grid.density, nx, ny, nz, smooth_sigma);

    // Scale the MC iso-threshold to account for the blur spreading density.
    // After a normalized 3-D separable Gaussian blur with 1-D center weight k0,
    // a voxel with raw density D contributes k0³·D to itself.  To keep the
    // `threshold` slider in "raw points/voxel" units we scale it accordingly.
    let mc_threshold = if smooth_sigma >= 0.5 {
        let radius = (3.0 * smooth_sigma).ceil() as usize;
        let size = 2 * radius + 1;
        let k_sum: f32 = (0..size)
            .map(|i| {
                let x = i as f32 - radius as f32;
                (-0.5 * x * x / (smooth_sigma * smooth_sigma)).exp()
            })
            .sum();
        let k0 = 1.0_f32 / k_sum;
        threshold * k0 * k0 * k0
    } else {
        threshold
    };

    // ── 5. Marching cubes ───────────────────────────────────────────────────
    let mc = MarchingCubes::new(
        (nx, ny, nz),
        (vs, vs, vs),
        (1.0, 1.0, 1.0),
        LinVec3::new(mn[0], mn[1], mn[2]),
        blurred_density,
        mc_threshold,
    )
    .ok()?;

    let mesh = mc.generate(MeshSide::OutsideOnly);

    if mesh.indices.is_empty() {
        return None;
    }

    // ── 6. Build output vertices ─────────────────────────────────────────────
    let mut vertices: Vec<BundleMeshVertex> = mesh
        .vertices
        .iter()
        .map(|v| {
            let wx = v.posit.x;
            let wy = v.posit.y;
            let wz = v.posit.z;
            let gx = (wx - mn[0]) / vs;
            let gy = (wy - mn[1]) / vs;
            let gz = (wz - mn[2]) / vs;

            let nv = v.normal;
            let len = (nv.x * nv.x + nv.y * nv.y + nv.z * nv.z).sqrt().max(1e-6);

            BundleMeshVertex {
                position: [wx, wy, wz],
                normal: [nv.x / len, nv.y / len, nv.z / len],
                color: color_strategy_for_point(
                    color_strategy,
                    &grid,
                    boundary_field,
                    Vec3::new(wx, wy, wz),
                    gx,
                    gy,
                    gz,
                ),
            }
        })
        .collect();

    let raw_indices: Vec<u32> = mesh.indices.iter().map(|&i| i as u32).collect();

    // ── 7. Filter connected components by enclosed volume ───────────────────
    let min_component_volume_mm3 = min_component_volume_mm3.max(0.0);
    let indices = connected_components(&vertices, &raw_indices)
        .into_iter()
        .filter(|component| component.volume_mm3 >= min_component_volume_mm3)
        .flat_map(|component| component.indices)
        .collect::<Vec<_>>();

    if indices.is_empty() {
        return None;
    }

    weld_and_recompute_normals(&mut vertices, &indices);

    Some(BundleMesh { vertices, indices })
}

#[cfg(test)]
mod tests {
    use super::{
        BundleMeshVertex, TAUBIN_SMOOTHING_ITERS, apply_taubin_smoothing, component_volume_mm3,
        connected_components, group_neighbors, welded_vertex_groups,
    };
    use glam::Vec3;

    fn vertex(position: [f32; 3]) -> BundleMeshVertex {
        BundleMeshVertex {
            position,
            normal: [0.0, 0.0, 1.0],
            color: [0.5, 0.5, 0.5, 1.0],
        }
    }

    #[test]
    fn taubin_smoothing_leaves_isolated_vertices_unchanged() {
        let vertices = vec![vertex([0.0, 0.0, 0.0]), vertex([1.0, 0.0, 0.0])];
        let (vertex_group, mut group_positions, _) = welded_vertex_groups(&vertices);
        let neighbors = group_neighbors(&vertex_group, &[], group_positions.len());
        let before = group_positions.clone();
        apply_taubin_smoothing(&mut group_positions, &neighbors);
        assert_eq!(before, group_positions);
    }

    #[test]
    fn taubin_smoothing_reduces_single_vertex_spike() {
        let vertices = vec![
            vertex([0.0, 0.0, 0.0]),
            vertex([1.0, 0.0, 0.0]),
            vertex([1.0, 1.0, 0.0]),
            vertex([0.0, 1.0, 0.0]),
            vertex([0.5, 0.5, 0.75]),
        ];
        let indices = vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
        let (vertex_group, mut group_positions, _) = welded_vertex_groups(&vertices);
        let neighbors = group_neighbors(&vertex_group, &indices, group_positions.len());
        let original_height = group_positions[4].z;

        apply_taubin_smoothing(&mut group_positions, &neighbors);

        assert_eq!(TAUBIN_SMOOTHING_ITERS, 4);
        assert!(group_positions[4].z < original_height);
        let centroid = group_positions
            .iter()
            .copied()
            .fold(Vec3::ZERO, |acc, p| acc + p)
            / group_positions.len() as f32;
        assert!(centroid.z > 0.0);
    }

    #[test]
    fn connected_components_preserve_disconnected_meshes() {
        let vertices = vec![
            vertex([0.0, 0.0, 0.0]),
            vertex([1.0, 0.0, 0.0]),
            vertex([0.0, 1.0, 0.0]),
            vertex([5.0, 5.0, 0.0]),
            vertex([6.0, 5.0, 0.0]),
            vertex([5.0, 6.0, 0.0]),
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];
        let components = connected_components(&vertices, &indices);
        assert_eq!(components.len(), 2);
        assert_eq!(components[0].indices.len(), 3);
        assert_eq!(components[1].indices.len(), 3);
    }

    #[test]
    fn component_volume_uses_absolute_signed_volume() {
        let vertices = vec![
            vertex([0.0, 0.0, 0.0]),
            vertex([1.0, 0.0, 0.0]),
            vertex([0.0, 1.0, 0.0]),
            vertex([0.0, 0.0, 1.0]),
        ];
        let ccw = vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
        let cw = vec![0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2];
        let ccw_volume = component_volume_mm3(&vertices, &ccw);
        let cw_volume = component_volume_mm3(&vertices, &cw);
        assert!(ccw_volume > 0.0);
        assert!((ccw_volume - cw_volume).abs() < 1e-6);
    }
}

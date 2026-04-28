//! Exact 3-pass 1-D EDT (Saito–Toriwaki) used to dilate binary voxel masks.
//!
//! Shared between Hausdorff plan prep (limiting / seed / no-end masks) and
//! pyAFQ plan prep (waypoint / exclusion / endpoint distance tolerances).

/// Dilate `mask` by a radius expressed in voxels using an exact 3-pass 1-D
/// EDT (Saito–Toriwaki). Returns the binary dilation (distance ≤ radius).
pub(crate) fn dilate_mask(mask: &[u8], dims: [u32; 3], radius_vox: f32) -> Vec<u8> {
    let nx = dims[0] as usize;
    let ny = dims[1] as usize;
    let nz = dims[2] as usize;
    let n = nx * ny * nz;
    if n == 0 {
        return Vec::new();
    }

    const INF: u64 = u64::MAX / 4;
    let mut g2: Vec<u64> = Vec::with_capacity(n);

    // Pass along x: for each (y, z) row, compute 1-D squared distance to the
    // nearest set voxel in that row.
    for k in 0..nz {
        for j in 0..ny {
            let mut d: u64 = INF;
            for i in 0..nx {
                let idx = i + nx * (j + ny * k);
                if mask[idx] != 0 {
                    d = 0;
                } else if d != INF {
                    d = d.saturating_add(1);
                }
                g2.push(if d == INF { INF } else { d * d });
            }
            let row_start = nx * (j + ny * k);
            let mut d: u64 = INF;
            for i in (0..nx).rev() {
                let idx = row_start + i;
                if mask[idx] != 0 {
                    d = 0;
                } else if d != INF {
                    d = d.saturating_add(1);
                }
                let sq = if d == INF { INF } else { d * d };
                if sq < g2[idx] {
                    g2[idx] = sq;
                }
            }
        }
    }

    let mut tmp = vec![0u64; n];
    for k in 0..nz {
        for i in 0..nx {
            let mut f = vec![0u64; ny];
            for j in 0..ny {
                f[j] = g2[i + nx * (j + ny * k)];
            }
            let out = lower_envelope(&f);
            for j in 0..ny {
                tmp[i + nx * (j + ny * k)] = out[j];
            }
        }
    }
    let mut out = vec![0u64; n];
    for j in 0..ny {
        for i in 0..nx {
            let mut f = vec![0u64; nz];
            for k in 0..nz {
                f[k] = tmp[i + nx * (j + ny * k)];
            }
            let sqd = lower_envelope(&f);
            for k in 0..nz {
                out[i + nx * (j + ny * k)] = sqd[k];
            }
        }
    }

    let r2 = (radius_vox * radius_vox).max(0.0);
    let r2_u = if r2 > (u64::MAX / 4) as f32 {
        u64::MAX / 4
    } else {
        r2.ceil() as u64
    };
    out.into_iter()
        .map(|d2| if d2 <= r2_u { 1u8 } else { 0u8 })
        .collect()
}

/// Felzenszwalb–Huttenlocher 1-D squared-distance transform (lower envelope
/// of parabolas). Input `f[q]` is the already-squared distance contribution
/// from previous dimensions; returns the same.
fn lower_envelope(f: &[u64]) -> Vec<u64> {
    let n = f.len();
    if n == 0 {
        return Vec::new();
    }
    const INF: u64 = u64::MAX / 4;
    let inf_f = INF as f64;
    let fd = |q: usize| if f[q] >= INF { inf_f } else { f[q] as f64 };

    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let mut k: usize = 0;
    for q in 1..n {
        loop {
            let vk = v[k] as f64;
            let qf = q as f64;
            let s = ((fd(q) + qf * qf) - (fd(v[k]) + vk * vk)) / (2.0 * (qf - vk));
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }

    let mut out = vec![0u64; n];
    let mut k: usize = 0;
    for q in 0..n {
        let qf = q as f64;
        while z[k + 1] < qf {
            k += 1;
        }
        let vk = v[k];
        let vkf = vk as f64;
        let d = (qf - vkf) * (qf - vkf) + fd(vk);
        out[q] = if d >= inf_f { INF } else { d.round() as u64 };
    }
    out
}

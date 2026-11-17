"""Independent, conservative finite-volume solution of the effective thermal PDE.

This module uses NumPy/SciPy and does not import TensorFlow or the PINN. The FV
and PINN share physical inputs, not their numerical operators. Coordinates in
the public result are cell centers on the unit box. Temperature is rise in K.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized


@dataclass
class FVResult:
    centers: tuple
    times_s: np.ndarray
    rise_K: np.ndarray
    diagnostics: list
    runtime_s: float


def _footprint(rect, nx, ny):
    """Exact fraction of each bottom cell face covered by the rectangle."""
    x0, x1, y0, y1 = rect
    x = np.arange(nx) / nx
    y = np.arange(ny) / ny
    wx = np.maximum(0, np.minimum(x + 1 / nx, x1) - np.maximum(x, x0))
    wy = np.maximum(0, np.minimum(y + 1 / ny, y1) - np.maximum(y, y0))
    return (nx * wx[:, None]) * (ny * wy[None, :])


def _robin(h, k, d):
    # Wall-to-air film and center-to-wall half-cell in series.
    return h / (1 + h * d / (2 * k))


def finite_volume(cfg, times=(1, 5, 10, 30, 60), shape=(32, 16, 5), dt=0.25):
    """Backward Euler, cell-centered finite volumes with exact heater area.

    Uses constant orthotropic conductivity and volumetric heat capacity.
    Supports nonuniform output times by shortening the last step to each.
    Qstorage_W uses the backward difference in integrated cell enthalpy;
    the reported balance independently sums the boundary transfers.
    """
    start = perf_counter()
    shape = tuple(shape)
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times))
            or np.any(times <= 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be finite, positive and strictly increasing")
    if len(shape) != 3 or any(n < 2 or int(n) != n for n in shape) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("shape requires three integers >=2, and dt must be positive")
    shape = tuple(int(n) for n in shape)
    nx, ny, nz = shape
    dx, dy, dz = cfg.L / nx, cfg.W / ny, cfg.H / nz
    centers = tuple((np.arange(n) + .5) / n for n in shape)
    node = np.arange(np.prod(shape)).reshape(shape)
    diag = np.zeros(shape, dtype=float)
    rows, cols, data = [], [], []
    for axis, conductance in enumerate((cfg.k_xy * dy * dz / dx,
                                         cfg.k_xy * dx * dz / dy,
                                         cfg.k_z * dx * dy / dz)):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lower, upper = tuple(lower), tuple(upper)
        a, b = node[lower].ravel(), node[upper].ravel()
        rows.extend((a, b))
        cols.extend((b, a))
        data.extend((np.full(a.size, -conductance), np.full(a.size, -conductance)))
        diag[lower] += conductance
        diag[upper] += conductance

    area = dx * dy
    fcpu, fgpu = _footprint(cfg.cpu, nx, ny), _footprint(cfg.gpu, nx, ny)
    gc = _robin(cfg.h_side, cfg.k_xy, dx) * dy * dz
    gz = _robin(cfg.h_z, cfg.k_z, dz) * area
    bottom_g = gz * np.clip(1 - fcpu - fgpu, 0, 1)
    diag[[0, -1], :, :] += gc
    diag[:, :, 0] += bottom_g
    diag[:, :, -1] += gz
    rows.append(node.ravel())
    cols.append(node.ravel())
    data.append(diag.ravel())
    n = node.size
    matrix = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n, n)).tocsc()
    source = np.zeros(shape)
    source[:, :, 0] = area * (fcpu * cfg.q_cpu + fgpu * cfg.q_gpu)
    source = source.ravel()
    if not np.isclose(source.sum(), cfg.power, rtol=1e-10):
        raise ValueError("Heater powers, fluxes and rectangle areas are inconsistent")
    capacity = cfg.rho_cp * dx * dy * dz
    theta = np.zeros(n)
    snapshots, metrics = [], []
    clock = 0.
    cached = {}
    for target in times:
        while clock < target:
            remaining = target - clock
            proposed = min(dt, remaining)
            # Share factorizations when decimal checkpoints produce roundoff
            # variants of the same step; preserve genuinely very short steps.
            step = round(proposed, 12) if proposed > 1e-10 else proposed
            if step not in cached:
                mass = diags(np.full(n, capacity / step), format="csc")
                cached[step] = factorized(mass + matrix)
            prev = theta
            theta = cached[step](source + prev * (capacity / step))
            clock += step
            if target - clock <= 1e-10:
                clock = target
        field = theta.reshape(shape)
        snapshots.append(field.copy())
        qleft = gc * field[0].sum()
        qright = gc * field[-1].sum()
        qz = gz * field[:, :, -1].sum() + np.sum(bottom_g * field[:, :, 0])
        storage = capacity * (theta - prev).sum() / step
        metrics.append({
            "time_s": float(target), "Qin_W": float(source.sum()),
            "Qleft_W": float(qleft), "Qright_W": float(qright),
            "Qz_W": float(qz), "Qstorage_W": float(storage),
            "balance_error_W": float(source.sum() - qleft - qright - qz - storage),
            "stored_energy_J": float(theta.sum() * capacity),
            "FV_Tcpu_wall_mean_C": float(cfg.T_amb - 273.15
                + np.sum(fcpu * field[:, :, 0]) / fcpu.sum() + cfg.q_cpu * dz / (2 * cfg.k_z)),
            "FV_Tgpu_wall_mean_C": float(cfg.T_amb - 273.15
                + np.sum(fgpu * field[:, :, 0]) / fgpu.sum() + cfg.q_gpu * dz / (2 * cfg.k_z)),
        })
    return FVResult(centers, times, np.stack(snapshots), metrics, perf_counter() - start)


def reference_anchors(cfg, seed=28, shape=(24, 12, 4), count=3072, wall_labels=True):
    """Optional hybrid training labels sampled from a distinct, coarser FV grid.

    Labels cover volume points, reconstructed heater surfaces and both fin bases. No test-grid values
    or test-time checkpoints are used. Returns X=(x/L,y/W,z/H,t/t_end) and
    Celsius-valued labels (shape N,1).
    """
    fractions = np.array([.001, .004, .01, .025, .06, .13, .25, .40, .65, .85, .99])
    output_times = np.array([1., 5., 10., 30., 60.])
    coincide = np.any(np.isclose(cfg.t_end*fractions[:, None], output_times, atol=1.e-8), axis=1)
    fractions = fractions[~coincide]
    ref = finite_volume(cfg, cfg.t_end * fractions, shape, dt=min(.2, cfg.t_end / 100))
    rng = np.random.default_rng(seed)
    points = np.stack(np.meshgrid(*ref.centers, indexing="ij"), axis=-1).reshape(-1, 3)
    near = np.zeros(len(points), dtype=bool)
    for x0, x1, y0, y1 in (cfg.cpu, cfg.gpu):
        near |= ((points[:, 0] >= x0) & (points[:, 0] <= x1)
                 & (points[:, 1] >= y0) & (points[:, 1] <= y1))
    if count < 1:
        raise ValueError("anchor count must be positive")
    all_ids = np.arange(len(points))
    hot_ids = np.flatnonzero(near)
    if not hot_ids.size:
        hot_ids = all_ids
    which = np.r_[rng.choice(all_ids, (count + 1) // 2), rng.choice(hot_ids, count // 2)]
    rng.shuffle(which)
    clocks = rng.integers(len(fractions), size=count)
    X = np.column_stack((points[which], fractions[clocks]))
    rise = ref.rise_K.reshape(len(fractions), -1)[clocks, which]
    # Include physically reconstructed face temperatures. Cell-center-only
    # labels leave the hottest wall outside the teacher's training coordinates.
    # Sample exact die areas and the two end faces; the finer evaluation grid
    # and all five requested output times remain separate.
    nx, ny, nz = shape
    n_face = count // 8 if wall_labels else 0
    fields = ref.rise_K
    for group, name in enumerate(("cpu", "gpu", "x0", "x1")):
        sl = slice(group*n_face, (group+1)*n_face)
        if not n_face:
            break
        t_ids = clocks[sl]
        uv = rng.random((n_face, 2))
        if name in ("cpu","gpu"):
            xa,xb,ya,yb = getattr(cfg, name)
            xx = xa + (xb-xa)*uv[:,0]
            yy = ya + (yb-ya)*uv[:,1]
            X[sl,:3] = np.column_stack((xx, yy, np.zeros(n_face)))
            base = fields[t_ids, np.minimum((xx*nx).astype(int),nx-1),
                          np.minimum((yy*ny).astype(int),ny-1), 0]
            q = cfg.q_cpu if name == "cpu" else cfg.q_gpu
            rise[sl] = base + q*cfg.H/(2*nz*cfg.k_z)
        else:
            X[sl,:3] = np.column_stack((np.full(n_face, 0. if name=="x0" else 1.), uv))
            base = fields[t_ids, 0 if name=="x0" else -1,
                          (uv[:,0]*ny).astype(int), (uv[:,1]*nz).astype(int)]
            rise[sl] = base/(1 + cfg.h_side*cfg.L/(2*nx*cfg.k_xy))
    return X.astype("float32"), (cfg.T_amb - 273.15 + rise[:, None]).astype("float32")

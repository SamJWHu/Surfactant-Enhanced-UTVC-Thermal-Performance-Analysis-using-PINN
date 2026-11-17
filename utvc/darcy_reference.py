"""Independent, cell-centered check of the sealed vapor Darcy closure.

The temperature-dependent interphase mass source is shared with the PINN.
This verifies the hydrodynamic solve conditional on that source, and does not
verify the source, vapor properties, or Darcy constitutive law experimentally.
"""
import numpy as np
from scipy.fft import dctn, idctn


def solve_vapor_darcy(cfg, source):
    """Gauge-free pressure [Pa] and center velocity [m/s] for j [kg/m³/s].

    Solve -Laplace(p) = mu_v*j / (rho_v * vapor_volume_fraction * K_v)
    with zero normal flux on all six walls, on an equal-spaced Cartesian
    finite-volume mesh. Remove the constant source mode required by the
    Neumann compatibility condition. Report that physical correction separately.
    """
    source = np.asarray(source, dtype=float)
    if source.ndim != 3 or min(source.shape) < 2 or not np.isfinite(source).all():
        raise ValueError('Vapor source must be finite on a 3-D grid with >=2 cells per axis.')
    dims = (cfg.L, cfg.W, cfg.H)
    ds = np.asarray(dims) / np.asarray(source.shape)
    vapor_mass_per_volume = cfg.rho_v * (1.0 - cfg.fill) * cfg.void_fraction
    mobility = cfg.vapor_permeability / cfg.mu_v
    removed_mean = float(source.mean())
    rhs = (source - removed_mean) / (vapor_mass_per_volume * mobility)
    eig = np.zeros(source.shape)
    for axis, (count, dx) in enumerate(zip(source.shape, ds)):
        mode_shape = [1, 1, 1]
        mode_shape[axis] = count
        modes = (2.0 * np.sin(np.pi * np.arange(count) / (2 * count)) / dx)**2
        eig += modes.reshape(mode_shape)
    eig[0, 0, 0] = 1.0
    spectrum = dctn(rhs, type=2, norm='ortho') / eig
    spectrum[0, 0, 0] = 0.0
    pressure = idctn(spectrum, type=2, norm='ortho')
    velocity = []
    for axis, step in enumerate(ds):
        # Finite-volume face fluxes with zero normal velocity on either wall.
        pad = [(0, 0)] * 3
        pad[axis] = (1, 1)
        gradient = np.diff(np.pad(pressure, pad, mode='edge'), axis=axis) / step
        a, b = [slice(None)]*3, [slice(None)]*3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        velocity.append(-0.5 * mobility * (gradient[tuple(a)] + gradient[tuple(b)]))
    return pressure, np.stack(velocity, axis=-1), removed_mean


def vapor_darcy_check(cfg, phase_rate, pressure, velocity, *, reference=None):
    """Cross-check PINN gauges after subtracting mean; RMS in SI units."""
    if reference is None:
        reference = solve_vapor_darcy(cfg, phase_rate)
    pref, vref, mean_j = reference
    pressure, velocity = np.asarray(pressure, dtype=float), np.asarray(velocity, dtype=float)
    if pressure.shape != pref.shape or velocity.shape != vref.shape:
        raise ValueError('Compare vapor pressure (nx,ny,nz) and velocity (nx,ny,nz,3).')
    p_error = (pressure - pressure.mean()) - pref
    v_error_sq = ((velocity - vref)**2).sum(axis=-1)
    return {
        'darcy_reference_pressure_span_Pa': float(np.ptp(pref)),
        'darcy_pinn_pressure_span_Pa': float(np.ptp(pressure)),
        'darcy_pressure_gauge_aligned_RMSE_Pa': float(np.sqrt(np.mean(p_error**2))),
        'darcy_reference_speed_RMS_m_s': float(np.sqrt(np.mean((vref**2).sum(axis=-1)))),
        'darcy_velocity_vector_RMSE_m_s': float(np.sqrt(np.mean(v_error_sq))),
        'darcy_removed_net_source_kg_s': mean_j * cfg.volume,
    }

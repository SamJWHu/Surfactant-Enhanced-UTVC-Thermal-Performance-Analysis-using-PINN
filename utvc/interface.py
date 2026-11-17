"""Differentiable liquid–vapor jump conditions on *resolved*, moving interfaces.

SI units. The normal points from liquid to vapor, jumps are vapor minus liquid,
j>0 is evaporation and kappa=div_s(n) is +2/R for a spherical liquid droplet.
Inputs are one-sided continuum/CFD/PINN traces evaluated at the *same* points.
This operator is independent of the homogenized chamber model. Run its moving
planar-interface consistency example with: python -m utvc.interface
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import tensorflow as tf


@dataclass
class PhaseTrace:
    rho: object  # kg/m³
    velocity: object  # (...,3), m/s
    pressure: object  # absolute Pa if evaluating kinetics
    viscous_traction: object  # (...,3), tau @ n in Pa (both phases use same n)
    temperature: object  # K
    grad_temperature: object  # (...,3), K/m
    conductivity: object  # W/(m K)


@dataclass
class SurfaceTrace:
    normal: object  # (...,3), unit liquid -> vapor
    velocity: object  # (...,3), velocity of interface parametrization
    temperature: object  # K; varies in space/time
    mass_flux: object  # signed j [kg/(m² s)]
    sigma: object  # N/m
    curvature: object  # 1/m
    grad_sigma: object  # (...,3), N/m²; tangential projection is applied


@dataclass
class SoluteTrace:
    mass_fraction: object  # liquid bulk mass fraction, adjacent to interface
    grad_mass_fraction: object  # (...,3), 1/m
    diffusivity: object  # liquid diffusivity [m²/s]
    adsorbed_flux: object = 0.0  # liquid to surface [kg/(m² s)]


@dataclass
class AdsorbedLayer:
    concentration: object  # Gamma [kg/m²]
    material_rate: object  # D^Gamma Gamma/Dt following surface velocity [kg/(m² s)]
    surface_div_velocity: object  # div_s(u_surface) [1/s], includes curvature motion
    surface_laplacian: object  # Delta_s(Gamma) [kg/m⁴]
    surface_diffusivity: object  # m²/s
    capacity: object  # saturation coverage [kg/m²]
    k_ads: object  # m³/(kg s), with liquid mass concentration in kg/m³
    k_des: object  # 1/s


def _tensor(value, dtype):
    if tf.is_tensor(value) or isinstance(value, tf.Variable):
        return tf.cast(value, dtype)
    return tf.convert_to_tensor(value, dtype=dtype)


def _column(value, n, dtype):
    return tf.broadcast_to(tf.reshape(_tensor(value, dtype), [-1, 1]), [n, 1])


def _vector(value, n, dtype):
    return tf.broadcast_to(tf.reshape(_tensor(value, dtype), [-1, 3]), [n, 3])


def _dot(a, b):
    return tf.reduce_sum(a * b, axis=-1, keepdims=True)


def water_saturation_pa(T):
    """Water IAPWS SR1-86(1992) vapor pressure, differentiable on [273.16,Tc)."""
    T = tf.convert_to_tensor(T)
    tc = _tensor(647.096, T.dtype)
    tf.debugging.assert_greater_equal(T, _tensor(273.16, T.dtype))
    tf.debugging.assert_less(T, tc)
    tau = 1 - T / tc
    terms = (
        (-7.85951783, 1.0),
        (1.84408259, 1.5),
        (-11.7866497, 3.0),
        (22.6807411, 3.5),
        (-15.9618719, 4.0),
        (1.80122502, 7.5),
    )
    return 22.064e6 * tf.exp(tc / T * tf.add_n([a * tau**b for a, b in terms]))


def newtonian_viscous_traction(grad_u, viscosity, normal):
    """Incompressible Newtonian traction tau.n; grad_u[i,j] is du_i/dx_j."""
    G = tf.convert_to_tensor(grad_u)
    n = _tensor(normal, G.dtype)
    mu = tf.reshape(_tensor(viscosity, G.dtype), [-1, 1])
    return mu * tf.linalg.matvec(G + tf.linalg.matrix_transpose(G), n)


def jump_residuals(
    liquid,
    vapor,
    surface,
    h_fg,
    *,
    solute=None,
    layer=None,
    accommodation=None,
    water_activity=1.0,
    vapor_water_partial_pa=None,
    dtype=tf.float64,
):
    """Return dimensional per-point residuals; each quantity must approach zero.

    Low-Mach Stefan balance neglects kinetic-energy differences, viscous work and
    surface-energy storage. Thermal contact is ideal (continuous one-sided T);
    a kinetic resistance can still produce departure from saturation pressure.
    Hertz–Knudsen uses the water-partial vapor pressure, unit water activity
    unless measured otherwise, common T_i, and equal evaporation/condensation
    accommodation. Supply water partial pressure separately for noncondensable
    gas analysis; the present pure-vapor demo uses total vapor pressure.
    """
    # A common flat-patch normal is often supplied once for many CFD samples.
    vectors = [
        surface.normal,
        surface.velocity,
        surface.grad_sigma,
        liquid.velocity,
        vapor.velocity,
        liquid.grad_temperature,
        vapor.grad_temperature,
        liquid.viscous_traction,
        vapor.viscous_traction,
    ]
    scalars = [
        surface.temperature,
        surface.mass_flux,
        surface.curvature,
        surface.sigma,
        liquid.rho,
        vapor.rho,
        liquid.temperature,
        vapor.temperature,
        liquid.pressure,
        vapor.pressure,
        liquid.conductivity,
        vapor.conductivity,
        h_fg,
    ]
    if solute is not None:
        vectors.append(solute.grad_mass_fraction)
        scalars.extend([solute.mass_fraction, solute.diffusivity, solute.adsorbed_flux])
    if layer is not None:
        scalars.extend(vars(layer).values())
    sizes = [tf.size(_tensor(x, dtype)) // 3 for x in vectors]
    sizes.extend(tf.size(_tensor(x, dtype)) for x in scalars)
    n = tf.reduce_max(tf.stack(sizes))

    def col(x):
        return _column(x, n, dtype)

    def vec(x):
        return _vector(x, n, dtype)

    normal = vec(surface.normal)
    tf.debugging.assert_near(
        _dot(normal, normal),
        tf.ones([n, 1], dtype=dtype),
        atol=_tensor(2.0e-5, dtype),
        message="Normal must be unit length",
    )
    us, ul, uv = vec(surface.velocity), vec(liquid.velocity), vec(vapor.velocity)
    j = col(surface.mass_flux)
    tl = vec(liquid.viscous_traction) - col(liquid.pressure) * normal
    tv = vec(vapor.viscous_traction) - col(vapor.pressure) * normal
    ds = vec(surface.grad_sigma)
    mar = ds - _dot(ds, normal) * normal
    ql = -col(liquid.conductivity) * _dot(vec(liquid.grad_temperature), normal)
    qv = -col(vapor.conductivity) * _dot(vec(vapor.grad_temperature), normal)
    ti = col(surface.temperature)
    r = {
        "liquid_mass_kg_m2_s": col(liquid.rho) * _dot(ul - us, normal) - j,
        "vapor_mass_kg_m2_s": col(vapor.rho) * _dot(uv - us, normal) - j,
        "momentum_Pa": tv
        - tl
        - j * (uv - ul)
        - col(surface.sigma) * col(surface.curvature) * normal
        + mar,
        "stefan_W_m2": ql - qv - j * col(h_fg),
        "liquid_contact_K": col(liquid.temperature) - ti,
        "vapor_contact_K": col(vapor.temperature) - ti,
    }
    if accommodation is not None:
        alpha = col(accommodation)
        aw = col(water_activity)
        tf.debugging.assert_positive(alpha)
        tf.debugging.assert_less_equal(alpha, tf.ones_like(alpha))
        tf.debugging.assert_positive(aw)
        pv_water = col(
            vapor.pressure if vapor_water_partial_pa is None else vapor_water_partial_pa
        )
        tf.debugging.assert_positive(pv_water)
        kinetic = (
            alpha
            * (aw * water_saturation_pa(ti) - pv_water)
            * tf.sqrt(_tensor(0.01801528 / (2 * math.pi * 8.314462618), dtype) / ti)
        )
        r["kinetic_kg_m2_s"] = j - kinetic
    if solute is not None:
        rl = col(liquid.rho)
        concentration = col(solute.mass_fraction)
        uptake = col(solute.adsorbed_flux)
        F = rl * (
            concentration * _dot(ul - us, normal)
            - col(solute.diffusivity) * _dot(vec(solute.grad_mass_fraction), normal)
        )
        r["solute_flux_kg_m2_s"] = F - uptake
        if layer is not None:
            gamma = col(layer.concentration)
            r["surface_transport_kg_m2_s"] = (
                col(layer.material_rate)
                + gamma * col(layer.surface_div_velocity)
                - col(layer.surface_diffusivity) * col(layer.surface_laplacian)
                - uptake
            )
            adsorption = (
                col(layer.k_ads) * rl * concentration * (col(layer.capacity) - gamma)
                - col(layer.k_des) * gamma
            )
            r["adsorption_kg_m2_s"] = uptake - adsorption
    elif layer is not None:
        raise ValueError(
            "Adsorbed layer balance requires liquid solute and adsorption flux traces."
        )
    return r


def interface_loss(residuals, scales):
    """Dimensionless MSE with explicit positive reference scales for each unit."""
    if set(scales) != set(residuals):
        raise ValueError(
            "Specify one dimensional reference scale for every jump residual."
        )
    terms = []
    for name, val in residuals.items():
        scale = _tensor(scales[name], val.dtype)
        tf.debugging.assert_positive(scale)
        terms.append(tf.reduce_mean(tf.square(val / scale)))
    return tf.add_n(terms)


def manufactured_traces(time_s, j, dtype=tf.float64):
    """Local moving-plane manufactured traces; j may be positive or negative.

    Temperature profiles on both sides are linear in signed interface distance
    and generate the gradients below. These enforce local jumps by construction;
    they do not constitute a closed solution of the chamber's bulk PDEs.
    """
    t = tf.reshape(_tensor(time_s, dtype), [-1, 1])
    j = tf.broadcast_to(tf.reshape(_tensor(j, dtype), [-1, 1]), tf.shape(t))
    z = tf.zeros_like(t)
    normal = tf.concat([z, z, tf.ones_like(t)], axis=1)
    rl, rv, hfg = 983.2, 0.1304, 2.358e6
    ti = 333.15 + 0.04 * t  # K
    alpha = 0.2
    kinetic_coeff = alpha * tf.sqrt(
        _tensor(0.01801528 / (2 * math.pi * 8.314462618), dtype) / ti
    )
    pv = water_saturation_pa(ti) - j / kinetic_coeff
    pl = pv + j**2 * (1 / rv - 1 / rl)  # recoil
    V = -j / rl  # stationary liquid
    us = V * normal
    uv = (V + j / rv) * normal
    q_v = 2.0  # W/m² in direction of normal
    q_l = j * hfg + q_v
    D = 5e-10
    cl = 0.003
    k_ads = (2e-9 + 1e-9) / (rl * cl * 1e-6)
    A = k_ads * rl * cl
    k_des = 0.001
    cap = 2e-6
    equilibrium = A * cap / (A + k_des)
    gamma = equilibrium + (1e-6 - equilibrium) * tf.exp(-(A + k_des) * t)
    J = (A + k_des) * (equilibrium - gamma)
    layer = AdsorbedLayer(
        concentration=gamma,
        material_rate=J,
        surface_div_velocity=0.0,
        surface_laplacian=0.0,
        surface_diffusivity=1e-9,
        capacity=cap,
        k_ads=k_ads,
        k_des=k_des,
    )
    solute = SoluteTrace(cl, (j * cl - J) / (rl * D) * normal, D, J)
    liquid = PhaseTrace(
        rl,
        tf.zeros_like(normal),
        pl,
        tf.zeros_like(normal),
        ti,
        -q_l / 0.654 * normal,
        0.654,
    )
    vapor = PhaseTrace(
        rv, uv, pv, tf.zeros_like(normal), ti, -q_v / 0.025 * normal, 0.025
    )
    surf = SurfaceTrace(normal, us, ti, j, 0.0662, 0.0, tf.zeros_like(normal))
    position = 0.0008 + V * t
    return liquid, vapor, surf, solute, layer, hfg, alpha, position


def main():
    parser = argparse.ArgumentParser(
        description="Manufactured planar evaporation / condensation jump verification."
    )
    parser.add_argument(
        "--csv", type=Path, help="Optional dimensional residuals and position CSV."
    )
    args = parser.parse_args()
    rows = []
    times = [1.0, 5.0, 10.0, 30.0, 60.0]
    for j in (0.00015, -0.00015):
        li, va, s, c, layer, hfg, alpha, position = manufactured_traces(times, j)
        r = jump_residuals(li, va, s, hfg, solute=c, layer=layer, accommodation=alpha)
        for i, t in enumerate(times):
            row = {
                "time_s": t,
                "j_kg_m2_s": j,
                "interface_z_mm": 1000 * float(position[i, 0]),
                "interface_T_K": float(s.temperature[i, 0]),
            }
            for k, term in r.items():
                row[k + "_max_abs"] = float(tf.reduce_max(abs(term[i])))
            rows.append(row)
    print(
        "Manufactured moving planar traces (evaporation and condensation, five times)"
    )
    for key in rows[0]:
        if key.endswith("_max_abs"):
            print(f"{key}: {max(r[key] for r in rows):.3e}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

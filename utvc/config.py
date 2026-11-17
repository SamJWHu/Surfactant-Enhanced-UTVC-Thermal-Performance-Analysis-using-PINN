"""SI inputs, case provenance, geometry checks and stratified boundary sampling.

Pure-water values are rounded, constant reference properties near 60 °C (ASHRAE
Handbook, Fundamentals, Ch. 1, Tables 3 and 6; NISTIR 5078). Effective plate
coefficients and wick/interfacial parameters are *example model inputs*.
Unknown mixture properties inherit water proxies. An additive name alone never
changes conductivity, viscosity, wetting or phase-change performance.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

# Percentage points, so the 0.30 wt% PEG case has c0=0.003, not 0.30.
FORMULATIONS = {
    "water": ("DI water", 0.0),
    "PEG600_0.30": ("PEG600", 0.30),
    "PEG600_0.40": ("PEG600", 0.40),
    "PEG1000_0.20": ("PEG1000", 0.20),
    "PEG1000_0.30": ("PEG1000", 0.30),
    "PPG600_0.20": ("PPG600", 0.20),
    "PPG600_0.30": ("PPG600", 0.30),
    "PPG1000_0.10": ("PPG1000", 0.10),
    "PPG1000_0.20": ("PPG1000", 0.20),
    "APG_C12C13_0.10": ("C12–C13 APG", 0.10),
    "APG_C12C13_0.15": ("C12–C13 APG", 0.15),
}
LOADS = {"80W": (30.0, 50.0), "150W": (35.0, 115.0)}
# These row-wise inputs are specific to a fluid or a particular assembled unit.
# Wall dimensions and sink conditions belong in the geometry/run configuration.
CAL_FIELDS = (
    "rho",
    "mu",
    "cp",
    "k_liq",
    "D",
    "rho_v",
    "mu_v",
    "h_fg",
    "sigma",
    "contact_angle_deg",
    "pore_radius",
    "permeability",
    "vapor_permeability",
    "interface_h_vol",
    "k_xy",
    "k_z",
    "rho_cp",
)


def cases():
    return tuple(FORMULATIONS)


def water_psat(T_K):
    """Water saturation pressure [Pa], IAPWS SR1-86(1992), Eq. (1)."""
    if not 273.16 <= T_K < 647.096:
        raise ValueError(
            "Water saturation correlation requires 273.16 <= T < 647.096 K."
        )
    v = 1.0 - T_K / 647.096
    coeffs = (-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502)
    powers = (1, 1.5, 3, 3.5, 4, 7.5)
    return 22.064e6 * math.exp(
        647.096 / T_K * sum(a * v**p for a, p in zip(coeffs, powers))
    )


@dataclass(frozen=True)
class Config:
    case: str = "water"
    load: str = "80W"
    L: float = 0.064
    W: float = 0.024
    H: float = 0.0016
    t_end: float = 60.0
    T_amb: float = 298.15
    dT: float = 40.0
    cpu_length: float = 0.020
    cpu_width: float = 0.010
    gpu_length: float = 0.010
    gpu_width: float = 0.010
    cpu_x: float = 0.375  # normalized centers
    cpu_y: float = 0.500
    gpu_x: float = 0.625
    gpu_y: float = 0.500
    fill: float = 0.5  # fraction of accessible internal void initially liquid
    wall: float = 0.00020  # copper wall thickness on each broad face
    wick_depth: float = 0.00025  # combined thickness of porous wick layers
    porosity: float = 0.55  # wick porosity; pillars not modeled separately
    rho_cu: float = 8960.0
    cp_cu: float = 385.0
    rho: float = 983.2  # kg/m³
    cp: float = 4185.0  # J/(kg K)
    mu: float = 0.000466  # Pa s
    rho_v: float = 0.1304  # kg/m³
    cp_v: float = 2000.0
    mu_v: float = 1.08e-5
    h_fg: float = 2.358e6  # J/kg
    k_liq: float = 0.654  # liquid only [W/(m K)]
    k_xy: float = 3000.0  # effective plate, in-plane [W/(m K)]; illustrative
    k_z: float = 80.0  # effective plate, through-plane [W/(m K)]; illustrative
    D: float = 5e-10  # tracer diffusivity [m²/s]; placeholder for blends
    sigma: float = 0.0662  # representative water [N/m]
    contact_angle_deg: float = 30.0  # hypothetical actual Cu wick advancing angle
    pore_radius: float = 10e-6
    permeability: float = 2e-11  # effective liquid/wick permeability [m²]
    interface_h_vol: float = 5e5  # lumped h_i A_i/V [W/(m³ K)]
    h_side: float = 45_000.0  # fin-base effective convection [W/(m² K)]
    h_z: float = 10.0  # external ambient convection, unloaded z faces
    # Optional measured overrides. Otherwise derive from fill and void geometry.
    vapor_K_override: float | None = None
    rho_cp_override: float | None = None
    source: str = ""
    measured_fields: tuple[str, ...] = ()

    def __post_init__(self):
        if self.case not in FORMULATIONS or self.load not in LOADS:
            raise ValueError(f"Unknown case/load {self.case!r}, {self.load!r}.")
        zero_allowed = {
            "h_z",
            "wick_depth",
            "wall",
            "contact_angle_deg",
            "cpu_x",
            "gpu_x",
            "cpu_y",
            "gpu_y",
        }
        for field in fields(self):
            if field.name in {"case", "load", "source", "measured_fields"}:
                continue
            val = getattr(self, field.name)
            if val is None and field.name in {"vapor_K_override", "rho_cp_override"}:
                continue
            if (
                not math.isfinite(val)
                or val < 0
                or (val == 0 and field.name not in zero_allowed)
            ):
                raise ValueError(f"Invalid {field.name} = {val}.")
        if not 0.0 < self.fill < 1.0 or not 0 < self.porosity <= 1.0:
            raise ValueError("Require 0 < fill < 1 and 0 < wick porosity <= 1.")
        if self.H <= 2 * self.wall + self.wick_depth:
            raise ValueError(
                "No vapor core fits between the specified copper walls and wick."
            )
        if self.contact_angle_deg > 180:
            raise ValueError("contact_angle_deg must be between 0 and 180.")
        water_psat(self.T_amb)
        for label, rect in (("CPU", self.cpu), ("GPU", self.gpu)):
            a, b, c, d = rect
            if not (0 <= a < b <= 1 and 0 <= c < d <= 1):
                raise ValueError(
                    f"{label} footprint exceeds chamber. Check dimensions and position."
                )
        a, b, c, d = self.cpu
        e, f, g, h = self.gpu
        if min(b, f) > max(a, e) and min(d, h) > max(c, g):
            raise ValueError(
                "Footprints overlap. Enlarge the chamber or change centers/die dimensions."
            )
        if self.A_top <= self.cpu_area + self.gpu_area:
            raise ValueError("No free z=0 face left outside CPU/GPU patches.")

    def _rectangle(self, cx, cy, length, width):
        dx, dy = 0.5 * length / self.L, 0.5 * width / self.W
        return (cx - dx, cx + dx, cy - dy, cy + dy)

    @property
    def cpu(self):
        return self._rectangle(self.cpu_x, self.cpu_y, self.cpu_length, self.cpu_width)

    @property
    def gpu(self):
        return self._rectangle(self.gpu_x, self.gpu_y, self.gpu_length, self.gpu_width)

    @property
    def P_cpu(self):
        return LOADS[self.load][0]

    @property
    def P_gpu(self):
        return LOADS[self.load][1]

    @property
    def power(self):
        return self.P_cpu + self.P_gpu

    @property
    def cpu_area(self):
        return self.cpu_length * self.cpu_width

    @property
    def gpu_area(self):
        return self.gpu_length * self.gpu_width

    @property
    def q_cpu(self):
        return self.P_cpu / self.cpu_area

    @property
    def q_gpu(self):
        return self.P_gpu / self.gpu_area

    @property
    def A_top(self):
        return self.L * self.W

    @property
    def volume(self):
        return self.A_top * self.H

    @property
    def c0(self):
        return FORMULATIONS[self.case][1] / 100.0

    @property
    def p_ref(self):
        return water_psat(self.T_amb)

    @property
    def void_fraction(self):
        """Internal void volume per total plate volume (neglect edge seals)."""
        return (self.H - 2 * self.wall - self.wick_depth * (1 - self.porosity)) / self.H

    @property
    def vapor_gap(self):
        # Liquid first occupies wick pores, then reduces the geometrical vapor gap.
        void_height = self.void_fraction * self.H
        core_height = self.H - 2 * self.wall - self.wick_depth
        return min(core_height, (1 - self.fill) * void_height)

    @property
    def vapor_permeability(self):
        """Ideal parallel-plate vapor gap closure; replace if externally measured."""
        if self.vapor_K_override is not None:
            return self.vapor_K_override
        return self.vapor_gap**2 / 12.0

    @property
    def rho_cp(self):
        """Illustrative volume-weighted wall/wick/liquid/vapor thermal storage."""
        if self.rho_cp_override is not None:
            return self.rho_cp_override
        fluid = (
            self.fill * self.rho * self.cp + (1 - self.fill) * self.rho_v * self.cp_v
        )
        return (
            1 - self.void_fraction
        ) * self.rho_cu * self.cp_cu + self.void_fraction * fluid

    @property
    def capillary_pressure(self):
        return (
            2
            * self.sigma
            * math.cos(math.radians(self.contact_angle_deg))
            / self.pore_radius
        )

    def dimensionless_groups(self, *, h_air=None, channel_Dh=None, k_air=None):
        """Characteristic diagnostics. Supply actual air-channel data for Nu.

        U_l and U_v assume all the applied power moves as latent heat through
        the homogenized accessible phase area; these scale factors are not
        model predictions. Re_L uses chamber length, not pore size. Bi uses
        plate conductivity while Nu uses the external fluid's conductivity.
        """
        liquid = self.fill * self.void_fraction
        vapor = (1 - self.fill) * self.void_fraction
        area = self.W * self.H
        ul = self.power / (self.rho * self.h_fg * liquid * area)
        uv = self.power / (self.rho_v * self.h_fg * vapor * area)
        alpha = self.k_liq / (self.rho * self.cp)
        groups = {
            "U_liquid_m_s": ul,
            "U_vapor_m_s": uv,
            "Re_liquid_L": self.rho * ul * self.L / self.mu,
            "Re_vapor_gap": self.rho_v * uv * self.vapor_gap / self.mu_v,
            "Pr_liquid": self.mu * self.cp / self.k_liq,
            "Pe_thermal_liquid_L": ul * self.L / alpha,
            "Pe_solute_L": ul * self.L / self.D,
            "Bi_fin_base_L": self.h_side * self.L / self.k_xy,
            "Bi_z_H": self.h_z * self.H / self.k_z,
            "Fo_x": self.k_xy * self.t_end / (self.rho_cp * self.L**2),
            "Fo_y": self.k_xy * self.t_end / (self.rho_cp * self.W**2),
            "Fo_z": self.k_z * self.t_end / (self.rho_cp * self.H**2),
        }
        air_data = (h_air, channel_Dh, k_air)
        if any(val is not None for val in air_data):
            if any(
                val is None or not math.isfinite(val) or val <= 0 for val in air_data
            ):
                raise ValueError(
                    "Air-channel Nu requires positive h_air, channel_Dh and k_air in SI."
                )
            groups["Nu_air_channel"] = h_air * channel_Dh / k_air
        return groups

    def audit(self):
        area_side = 2 * self.W * self.H
        area_all = 2 * (self.L * self.W + self.L * self.H + self.W * self.H)
        charge_g = 1000 * self.fill * self.void_fraction * self.volume * self.rho
        return {
            "case": self.case,
            "load": self.load,
            "wt_percent": 100 * self.c0,
            "fill": self.fill,
            "length_width_thickness_mm": [1000 * v for v in (self.L, self.W, self.H)],
            "cpu_gpu_flux_W_m2": [self.q_cpu, self.q_gpu],
            "cpu_gpu_integrated_W": [
                self.q_cpu * self.cpu_area,
                self.q_gpu * self.gpu_area,
            ],
            "condenser_fraction_total_external_area": area_side / area_all,
            "condenser_fraction_one_broad_face_area": area_side / self.A_top,
            "end_fin_UA_W_per_K": self.h_side * area_side,
            "capillary_pressure_budget_Pa": self.capillary_pressure,
            "charge_mass_g": charge_g,
            "additive_mass_mg": 1000 * charge_g * self.c0,
            "saturation_pressure_at_initial_ambient_Pa": self.p_ref,
            "source": self.source
            or "shared water proxies and illustrative plate/wick closures",
            "measured_fields": list(self.measured_fields),
            "characteristic_scales_and_groups": self.dimensionless_groups(),
        }


def _calibrated(cfg, csv_file):
    with Path(csv_file).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"case", "source"} <= set(
            reader.fieldnames
        ):
            raise ValueError("Calibration CSV requires case,source columns.")
        unknown = set(reader.fieldnames) - {"case", "source", *CAL_FIELDS}
        if unknown:
            raise ValueError(f"Unknown calibration column(s): {sorted(unknown)}")
        rows = list(reader)
    seen = set()
    for row in rows:
        name = (row["case"] or "").strip()
        if name not in FORMULATIONS or name in seen or None in row:
            raise ValueError(f"Unknown/duplicate/malformed calibration row: {name}")
        seen.add(name)
        values = {
            key: float(row[key]) for key in CAL_FIELDS if (row.get(key) or "").strip()
        }
        source = (row.get("source") or "").strip()
        if values and not source:
            raise ValueError(
                f"Record measurement/source ID for supplied values in {name}."
            )
        if name == cfg.case and values:
            kwargs = {
                {
                    "vapor_permeability": "vapor_K_override",
                    "rho_cp": "rho_cp_override",
                }.get(k, k): v
                for k, v in values.items()
            }
            cfg = replace(cfg, source=source, measured_fields=tuple(values), **kwargs)
    return cfg


def build_config(case="water", load="80W", fill=0.5, calibration=None, **overrides):
    """Build a run; direct design overrides are SI. Calibration file values win."""
    cfg = Config(case=case, load=load, fill=fill, **overrides)
    return _calibrated(cfg, calibration) if calibration else cfg


def _open_unit(rng, n):
    # Avoid t=0 on the switched-on heater BC, and intersections of two faces.
    return np.clip(rng.random((n, 4), dtype=np.float32), 1e-6, 1 - 1e-6)


def _free_bottom(cfg, rng, n):
    # Partition the entire face along patch edges, then sample remaining tiles
    # proportional to area. This remains efficient for almost-covered faces.
    x_edges = sorted({0.0, 1.0, *cfg.cpu[:2], *cfg.gpu[:2]})
    y_edges = sorted({0.0, 1.0, *cfg.cpu[2:], *cfg.gpu[2:]})
    free = []
    weights = []
    for xl, xh in zip(x_edges, x_edges[1:]):
        for yl, yh in zip(y_edges, y_edges[1:]):
            x, y = (xl + xh) / 2, (yl + yh) / 2
            if any(a < x < b and c < y < d for a, b, c, d in (cfg.cpu, cfg.gpu)):
                continue
            free.append((xl, xh, yl, yh))
            weights.append((xh - xl) * (yh - yl))
    weights = np.array(weights)
    rectangles = np.array(free)[rng.choice(len(free), n, p=weights / weights.sum())]
    points = _open_unit(rng, n)
    a, b, c, d = rectangles.T
    points[:, 0] = a + points[:, 0] * (b - a)
    points[:, 1] = c + points[:, 1] * (d - c)
    points[:, 2] = 0.0
    return points


def sample_points(cfg, n_int, n_bc, n_ic, seed):
    """Unit-box coordinates (x/L,y/W,z/H,t/t_end); n_bc is per BC patch."""
    if min(n_int, n_bc, n_ic) < 1:
        raise ValueError(
            "Number of interior, boundary and initial points must be positive."
        )
    rng = np.random.default_rng(seed)
    out = {"interior": _open_unit(rng, n_int), "initial": _open_unit(rng, n_ic)}
    out["initial"][:, 3] = 0.0
    for name, axis, coordinate in (
        ("x0", 0, 0.0),
        ("x1", 0, 1.0),
        ("y0", 1, 0.0),
        ("y1", 1, 1.0),
        ("z1", 2, 1.0),
    ):
        face = _open_unit(rng, n_bc)
        face[:, axis] = coordinate
        out[name] = face
    out["z0"] = _free_bottom(cfg, rng, n_bc)
    for name, (a, b, c, d) in (("cpu", cfg.cpu), ("gpu", cfg.gpu)):
        patch = _open_unit(rng, n_bc)
        patch[:, 0] = a + (b - a) * patch[:, 0]
        patch[:, 1] = c + (d - c) * patch[:, 1]
        patch[:, 2] = 0.0
        out[name] = patch
    return out

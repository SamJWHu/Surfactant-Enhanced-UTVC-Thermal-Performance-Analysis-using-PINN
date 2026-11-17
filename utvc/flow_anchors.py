"""Optional coarse vapor Darcy guidance for the reduced flow head.

The phase closure separates as j(x,t) = mdot(t) * profile(x) / V. Linearity of
constant-property Darcy flow makes one coarse geometry solve sufficient. The
PINN still predicts its own flow field when this teacher is disabled or absent
at inference. This module never adjusts the heat equation or its phase source.
"""
import numpy as np
import tensorflow as tf

from .darcy_reference import solve_vapor_darcy


class VaporAnchors:
    def __init__(self, cfg, model, shape=(16, 10, 4)):
        self.shape = tuple(shape)
        centers = [(np.arange(n) + .5) / n for n in shape]
        coords = np.stack(np.meshgrid(*centers, indexing='ij'), axis=-1).reshape(-1, 3)
        phase_points = np.column_stack([coords, np.ones(len(coords))])
        mdot_ref = cfg.power / cfg.h_fg
        profile = model.phase_profile(tf.constant(phase_points, tf.float32)).numpy().reshape(shape)
        p, v, _ = solve_vapor_darcy(cfg, profile * mdot_ref / cfg.volume)
        grid = np.concatenate([p[..., None], v], axis=-1)
        self.grid = tf.constant(grid, tf.float32)
        # The PINN gauges vapor pressure to zero at the chamber center at each t.
        self.p_center = self.interpolate(tf.constant([[.5, .5, .5]], tf.float32))[0, 0]
        self.p_scale = float(np.sqrt(np.mean((p - self.p_center.numpy())**2)))
        self.v_scale = float(np.sqrt(np.mean(np.sum(v**2, axis=-1))))
        self.mdot_ref = mdot_ref
        # An absolute floor keeps the t->0 comparison well conditioned without
        # clipping the signed circulation used to generate actual targets.
        early = tf.constant([[.5, .5, .5, 1.]], tf.float32)
        _, _, mature_mdot = model._phase(early)
        self.scale_floor = max(1.e-6, .25 * abs(float(mature_mdot[0, 0])) / mdot_ref)

    def interpolate(self, points):
        """Trilinear cell-center interpolation with clamped exterior half-cells."""
        size = tf.constant(self.shape, tf.float32)
        grid_coordinate = tf.clip_by_value(points[:, :3] * size - .5, 0., size - 1.)
        lower = tf.cast(tf.floor(grid_coordinate), tf.int32)
        frac = grid_coordinate - tf.cast(lower, tf.float32)
        high = tf.minimum(lower + 1, tf.constant(self.shape, tf.int32) - 1)
        result = tf.zeros([tf.shape(points)[0], 4], tf.float32)
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    pick = tf.constant([bool(i), bool(j), bool(k)])
                    idx = tf.where(pick, high, lower)
                    weight = tf.reduce_prod(tf.where(pick, frac, 1. - frac), axis=1, keepdims=True)
                    result += weight * tf.gather_nd(self.grid, idx)
        return result

    def losses(self, model, points):
        points = tf.cast(points, tf.float32)
        _, _, mdot = model._phase(points)
        amplitude = tf.stop_gradient(mdot / self.mdot_ref)
        denom = tf.maximum(tf.abs(amplitude), self.scale_floor)
        table = self.interpolate(points)
        target_p = amplitude * (table[:, 0:1] - self.p_center)
        target_v = amplitude * table[:, 1:4]
        state = model._states(points)
        pred_p = model.P_v * state[8]
        pred_v = model.Uv * tf.concat(state[4:7], axis=1) * tf.constant(model.anis, points.dtype)
        dp = (pred_p - target_p) / (denom * self.p_scale)
        dv = (pred_v - target_v) / (denom * self.v_scale)
        return tf.reduce_mean(dp**2), tf.reduce_mean(tf.reduce_sum(dv**2, axis=1))

import unittest
from dataclasses import replace
import numpy as np
import tensorflow as tf

from utvc.interface import (
    manufactured_traces,
    jump_residuals,
    interface_loss,
    newtonian_viscous_traction,
)


class JumpTests(unittest.TestCase):
    def test_manufactured_evaporation_and_condensation(self):
        for j in (1.5e-4, -1.5e-4):
            li, va, s, c, layer, hfg, alpha, z = manufactured_traces(
                [1.0, 5.0, 10.0, 30.0, 60.0], j
            )
            r = jump_residuals(
                li,
                va,
                replace(s, normal=[0.0, 0.0, 1.0]),
                hfg,
                solute=c,
                layer=layer,
                accommodation=alpha,
            )
            for name, values in r.items():
                with self.subTest(name=name):
                    np.testing.assert_allclose(values, 0, atol=2e-11, rtol=0)
            self.assertEqual(np.sign(float(z[-1, 0] - z[0, 0])), -np.sign(j))
            self.assertGreater(float(s.temperature[-1, 0]), float(s.temperature[0, 0]))

    def test_perturbed_conduction_flux_and_derivative(self):
        li, va, s, c, layer, hfg, alpha, z = manufactured_traces([1.0, 5.0], 1.5e-4)
        extra = tf.Variable(125.0, dtype=tf.float64)
        with tf.GradientTape() as tape:
            changed = replace(
                li,
                grad_temperature=li.grad_temperature
                - extra * s.normal / li.conductivity,
            )
            terms = jump_residuals(changed, va, s, hfg)
            loss = interface_loss(
                terms, {name: 100.0 if name == "stefan_W_m2" else 1.0 for name in terms}
            )
        np.testing.assert_allclose(terms["stefan_W_m2"], 125.0, atol=1e-11)
        self.assertAlmostEqual(
            float(tape.gradient(loss, extra)), 2 * 125.0 / 100.0**2, places=11
        )

    def test_recoil_curvature_marangoni_and_partial_pressure(self):
        li, va, s, c, layer, hfg, alpha, z = manufactured_traces([1.0, 5.0], 1.5e-4)
        # Liquid droplet p_l - p_v includes sigma*(2/R) in addition to recoil.
        curvature = 2000.0
        grad = tf.constant(
            [[0.035, 0.0, 50.0]], tf.float64
        )  # normal component drops out
        theta = tf.constant([[0.035, 0.0, 0.0]], tf.float64)
        li = replace(
            li,
            pressure=li.pressure + s.sigma * curvature + 10.0,
            viscous_traction=theta,
        )
        s = replace(s, curvature=curvature, grad_sigma=grad)
        va = replace(va, pressure=va.pressure + 10.0)
        r = jump_residuals(
            li,
            va,
            s,
            hfg,
            accommodation=alpha,
            vapor_water_partial_pa=va.pressure - 10.0,
        )
        np.testing.assert_allclose(r["momentum_Pa"], 0.0, atol=6e-12)
        np.testing.assert_allclose(r["kinetic_kg_m2_s"], 0.0, atol=1e-15)
        bad = jump_residuals(replace(li, viscous_traction=[0.0, 0.0, 0.0]), va, s, hfg)
        np.testing.assert_allclose(bad["momentum_Pa"][:, 0], 0.035, atol=1e-12)

    def test_simple_shear_traction(self):
        G = tf.constant(
            [[[0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], tf.float64
        )
        np.testing.assert_allclose(
            newtonian_viscous_traction(G, 0.002, [0.0, 0.0, 1.0]), [[0.006, 0.0, 0.0]]
        )


if __name__ == "__main__":
    unittest.main()

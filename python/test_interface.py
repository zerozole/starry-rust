import math
import unittest
from starry_rust import Map


class InterfaceTests(unittest.TestCase):
    def test_uniform_map_and_amplitude(self):
        m = Map(3)
        self.assertAlmostEqual(m.intensity(), 1/math.pi, places=14)
        self.assertAlmostEqual(m.flux(), 1.0, places=14)
        m.amp = 2.0
        self.assertAlmostEqual(m.flux(ro=.1), 1.98, places=13)

    def test_native_rotation(self):
        m = Map(1)
        m.y[3] = 1.0
        m.rotate([0, 0, 2], math.pi/2)
        for a, b in zip(m.y, [1, 1, 0, 0]):
            self.assertAlmostEqual(a, b, places=14)

    def test_native_gradient(self):
        m = Map()
        for a, b in zip(m.flux_gradient(ro=.2), [0., 0., -.4]):
            self.assertAlmostEqual(a, b, places=13)
        m.amp = 2.
        self.assertAlmostEqual(m.flux_gradient(ro=.2)[2], -.8, places=13)

    def test_limb_darkening(self):
        m = Map()
        u, r = .5, .3
        expected = ((1-u)*(1-r*r)+2*u/3*(1-r*r)**1.5)/(1-u/3)
        self.assertAlmostEqual(m.flux_limb_darkened([u], ro=r), expected, places=13)
        self.assertAlmostEqual(m.flux_limb_darkened([.4, .2]), 1., places=13)

    def test_invalid_arguments(self):
        for degree in [-1, 33, .5, True]:
            with self.assertRaises(ValueError):
                Map(degree)
        m = Map(1)
        with self.assertRaises(ValueError):
            m.flux(ro=-.1)
        with self.assertRaises(ValueError):
            m.intensity(lat=math.pi)
        with self.assertRaises(ValueError):
            m.rotate([0, 0, 0], 1.)
        with self.assertRaises(ValueError):
            m.flux_gradient(ro=1.)
        m.y = [1.]
        with self.assertRaises(ValueError):
            m.flux()


if __name__ == "__main__":
    unittest.main()

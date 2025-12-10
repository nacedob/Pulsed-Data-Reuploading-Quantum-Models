from icecream import ic
from src.utils import trace_out_dm, increase_dimensions
from src.Sampler import Sampler
import numpy as np
import unittest


class UtilsTest(unittest.TestCase):

    def test_recover_dm_random(self):
        """
        Take a separable state
        :return:
        """
        state0 = np.random.rand(2, 1)
        state0 = state0 / np.linalg.norm(state0)

        state1 = np.random.rand(2, 1)
        state1 = state1 / np.linalg.norm(state1)
        state = np.kron(state0, state1)

        dm = state @ state.T
        dm0 = state0 @ state0.T
        dm1 = state1 @ state1.T
        traceddm0 = trace_out_dm(dm, 0)
        traceddm1 = trace_out_dm(dm, 1)

        tolerance = 1e-6
        ones = np.ones((2, 2))
        ic(dm0, traceddm0, dm1, traceddm1)
        self.assertTrue((abs(dm0 - traceddm0) < tolerance * ones).all())
        self.assertTrue((abs(dm1 - traceddm1) < tolerance * ones).all())

    def test_increase_dimensions(self):
        points, labels = Sampler.circle(n_points=3)
        original_dimension = points.shape[1]
        self.assertEqual(2, original_dimension)

        transfomed_points = increase_dimensions(points, 5)
        transformed_dimension = transfomed_points.shape[1]
        self.assertEqual(5, transformed_dimension)

        # Check that the original data is preserved
        for i in range(points.shape[0]):
            self.assertTrue((points[i] == transfomed_points[i][:2]).all())
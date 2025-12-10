import unittest
from src.Sampler import Sampler
import numpy as np
from src.visualization import plot_2d_dataset
import matplotlib.pyplot as plt
import sys


class TestPlot2DDataset(unittest.TestCase):

    def test_circle(self):
        n_points = 500
        data, labels = Sampler.circle(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_2d_dataset(data, labels, title='circle')

    def test_stripes(self):
        n_points = 50000
        data, labels = Sampler.stripes(n_points=n_points, stripe_angle=np.pi/10)
        self.assertEqual(data.shape, (n_points, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_2d_dataset(data, labels, title='Stripes')

    def test_1_stripe(self):
        n_points = 50000
        data, labels = Sampler.stripes(n_points=n_points, n_stripes=1, stripe_angle=np.pi / 10)
        self.assertEqual(data.shape, (n_points, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_2d_dataset(data, labels, title='Stripes')

    def test_annulus(self):
        n_points = 5000
        data, labels = Sampler.annulus(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_2d_dataset(data, labels, title='annulus')

    def test_multi_circle(self):
        n_points = 1000
        data, labels = Sampler.multi_circle(n_points=n_points, radii=[0.3, 0.15], centers=[[-1,1], [1, -1]])
        self.assertEqual(data.shape, (n_points, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_2d_dataset(data, labels, title='multi_circle')

    def test_sinus(self):
        n_points = 10000
        data, labels = Sampler.sinus(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_2d_dataset(data, labels, title='sinus')

    def test_corners(self):
        n_points = 10000
        data, labels = Sampler.corners(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_2d_dataset(data, labels, title='corners')

    def test_spiral(self):
        n_points = 10000
        data, labels = Sampler.spiral(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_2d_dataset(data, labels, title='spiral')

    def test_rectangle(self):
        n_points = 1000
        data, labels = Sampler.rectangle(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_2d_dataset(data, labels, title='rectangle')

if __name__ == "__main__":
    unittest.main()

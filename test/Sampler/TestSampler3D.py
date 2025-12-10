import unittest
from src.Sampler import Sampler3D
import numpy as np
from src.visualization import plot_3d_dataset
import matplotlib.pyplot as plt
import sys


class TestPlot3DDataset(unittest.TestCase):

    def test_torus(self):
        n_points = 1500
        data, labels = Sampler3D.torus(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='Torus')

    def test_shell(self):
        n_points = 700
        data, labels = Sampler3D.shell(n_points=n_points)
        plot_3d_dataset(data, labels, title='Shell')
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization

    def test_pyramid(self):
        n_points = 1000
        data, labels = Sampler3D.pyramid(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='Pyramid')

    def test_cube(self):
        n_points = 1000
        data, labels = Sampler3D.cube(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='Cube')

    def test_multi_spheres(self):
        n_points = 1000
        data, labels = Sampler3D.multi_spheres(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='MultiSphere')

    def test_corners(self):
        n_points = 1000
        data, labels = Sampler3D.corners3d(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='Corners')

    def test_sinus(self):
        n_points = 5000
        data, labels = Sampler3D.sinus3d(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='Sinus')

    def test_cylinder(self):
        n_points = 1000
        data, labels = Sampler3D.cylinder(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='Cylinder')

    def test_ellipsoid(self):
        n_points = 1000
        data, labels = Sampler3D.ellipsoid(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='Ellipsoid')

    def test_helix(self):
        n_points = 1000
        data, labels = Sampler3D.helix(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='Helix')

    def test_butterfly(self):
        n_points = 1000
        data, labels = Sampler3D.butterfly(n_points=n_points)
        self.assertEqual(data.shape, (n_points, 3))
        self.assertEqual(len(np.unique(labels)), 2)
        # Visualization
        plot_3d_dataset(data, labels, title='butterfly')

if __name__ == "__main__":
    unittest.main()

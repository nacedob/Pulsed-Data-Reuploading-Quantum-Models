from src.Sampler import MNISTSampler
import unittest
import jax.numpy as jnp
from icecream import ic
from src.visualization import plot_3d_dataset, plot_2d_dataset



class MNISTSamplerTest(unittest.TestCase):

    def test_fashion(self):
        n_train = 2000
        n_test = 1000

        train_images, train_labels, test_images, test_labels = MNISTSampler.fashion(n_train=n_train, n_test=n_test)

        self.assertEqual(len(train_images), len(train_labels))
        self.assertEqual(len(test_images), len(test_labels))
        self.assertEqual(len(train_images), n_train)
        self.assertEqual(len(test_images), n_test)
        self.assertEqual(len(train_images[0]), 4)
        self.assertEqual(len(test_images[0]), 4)

        self.assertIsInstance(train_images, jnp.ndarray)
        self.assertIsInstance(train_images[0], jnp.ndarray)
        self.assertIsInstance(train_labels, jnp.ndarray)
        self.assertIsInstance(train_labels[0].item(), int)
        self.assertTrue(train_labels[0] in [0, 1])
        self.assertIsInstance(test_images, jnp.ndarray)
        self.assertIsInstance(test_images[0], jnp.ndarray)
        self.assertTrue(test_labels[0].item() in [0, 1])

    def test_fashion_visualization(self):
        data, labels, _, _ = MNISTSampler.fashion(n_train=1000, points_dimension=3)
        plot_3d_dataset(data, labels, title='fashion - 3D Plot Data')
        data, labels, _, _ = MNISTSampler.fashion(n_train=1000, points_dimension=2)
        plot_2d_dataset(data, labels, title='fashion - 2D Plot Data')

    def test_numbers(self):
        n_train = 2000
        n_test = 1000

        train_images, train_labels, test_images, test_labels = MNISTSampler.digits(n_train=n_train, n_test=n_test)

        self.assertEqual(len(train_images), len(train_labels))
        self.assertEqual(len(test_images), len(test_labels))
        self.assertEqual(len(train_images), n_train)
        self.assertEqual(len(test_images), n_test)
        self.assertEqual(len(train_images[0]), 4)
        self.assertEqual(len(test_images[0]), 4)

        self.assertIsInstance(train_images, jnp.ndarray)
        self.assertIsInstance(train_images[0], jnp.ndarray)
        self.assertIsInstance(train_labels, jnp.ndarray)
        self.assertIsInstance(train_labels[0].item(), int)
        self.assertTrue(train_labels[0] in [0, 1])
        self.assertIsInstance(test_images, jnp.ndarray)
        self.assertIsInstance(test_images[0], jnp.ndarray)
        self.assertTrue(test_labels[0].item() in [0, 1])

    def test_numbers_visualization(self):
        data, labels, _, _ = MNISTSampler.digits(n_train=1000, points_dimension=3)
        plot_3d_dataset(data, labels, title='numbers - 3D Plot Data')
        data, labels, _, _ = MNISTSampler.digits(n_train=1000, points_dimension=2)
        plot_2d_dataset(data, labels, title='numbers - 2D Plot Data')

if __name__ == '__main__':
    unittest.main()

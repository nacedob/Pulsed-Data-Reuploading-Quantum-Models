from warnings import warn
from icecream import ic
import pandas as pd
from numpy import load as np_load, savez_compressed as np_savez_compressed
try:
    from src.utils import get_current_folder
except ModuleNotFoundError:
    from src.utils import get_current_folder
from .utils import get_random_subset, reduce_dimension
from sklearn.datasets import load_iris, fetch_openml
import os
from jax import numpy as jnp
from pennylane import numpy as qnp
import random
from src.utils import get_root_path


def _load_and_filter_data(train_data, test_data, label1, label2):
    # Separar características (X) y etiquetas (y)
    x_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    x_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

    # Filtrar las filas que corresponden a las etiquetas deseadas
    train_filter = y_train.isin([label1, label2])
    test_filter = y_test.isin([label1, label2])

    # Aplicar el filtro
    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    # Convertir las etiquetas a 0 y 1
    y_train = y_train.apply(lambda y: 0 if y == label1 else 1)
    y_test = y_test.apply(lambda y: 0 if y == label1 else 1)

    # Normalizar los píxeles entre 0 y 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convertir a listas si es necesario
    x_train, y_train = x_train.values.tolist(), y_train.values.tolist()
    x_test, y_test = x_test.values.tolist(), y_test.values.tolist()

    return x_train, y_train, x_test, y_test


def process_dataset(x_train, y_train, x_test, y_test, points_dimension, interface):

    # Reduce x sets to 4 dimensions
    x_train = reduce_dimension(x_train, new_dim=points_dimension, feature_range=(-1, 1))
    if x_test:
        x_test = reduce_dimension(x_test, new_dim=points_dimension, feature_range=(-1, 1))

    if interface == 'jax':
        x_train = jnp.array(x_train)
        y_train = jnp.array(y_train)
        if x_test is not None:
            x_test = jnp.array(x_test)
            y_test = jnp.array(y_test)
    elif interface == 'pennylane':
        x_train = qnp.array(x_train, requires_grad=False)
        y_train = qnp.array(y_train, requires_grad=False)
        x_test = qnp.array(x_test, requires_grad=False)
        y_test = qnp.array(y_test, requires_grad=False)
    else:  # interface = normal numpy
        pass

    return x_train, y_train, x_test, y_test


class MNISTSampler:

    @staticmethod
    def fashion(n_train: int = 2000, n_test: int = 1000, points_dimension: int = 4,
                label1: int = 3, label2: int = 6, folder: str = None,
                seed: int = None, interface: str = 'jax') -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Retrieves and processes a subset of the MNIST dataset for binary classification between two specified labels.

        Args:
        n_train (int): Number of training samples to retrieve. Defaults to 2000.
        n_test (int): Number of test samples to retrieve. Defaults to 1000.
        label1 (int): The first label for binary classification. Defaults to 3 (# dress)
        label2 (int): The second label for binary classification. Defaults to 6 (# shirt)
        folder (str, optional): The folder path where the MNIST data files are located. If None, defaults to a 'mnist'
        folder in the current directory.
        interface (str, optional): The interface (of numpy) to get the data. Defaults to 'jax'. Available = 'jax',
        'pennylane' or 'numpy'

        Returns:
        x_train,
        y_train,
        x_test,
        y_test
        """

        if interface not in ['pennylane', 'jax', 'numpy']:
            raise ValueError(f"Invalid interface: {interface}. Available interfaces are 'jax', 'pennylane' or 'numpy'.")

        if seed is not None:
            random.seed(seed)

        if folder is None:
            folder = os.path.join(get_current_folder(), 'mnist')

        train_file = os.path.join(folder, 'fashion-mnist_train.csv')
        test_file = os.path.join(folder, 'fashion-mnist_test.csv')

        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        x_train, y_train, x_test, y_test = _load_and_filter_data(train_data, test_data, label1, label2)

        # Random subsets
        x_train, y_train, train_indices = get_random_subset(x_train, y_train, n_train, seed=seed)
        x_test, y_test, train_indices = get_random_subset(x_test, y_test, n_test, seed=seed,
                                                          exclude_indices=train_indices)

        dataset_processed = process_dataset(x_train, y_train, x_test, y_test, points_dimension, interface)
        return dataset_processed

    @staticmethod
    def digits_(n_train: int = 2000, n_test: int = 1000, points_dimension: int = 4,
                label1: int = 8, label2: int = 0,
                seed: int = None, interface: str = 'jax') -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Load the MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        mask = (y == label1) | (y == label2)
        X_filtered, y_filtered = X[mask], y[mask]
        y_filtered = jnp.where(y_filtered == label1, 0, 1)

        # Random subsets
        x_train, y_train, train_indices = get_random_subset(X_filtered, y_filtered, n_train, seed=seed)
        x_test, y_test, train_indices = get_random_subset(X_filtered, y_filtered, n_test, seed=seed,
                                                          exclude_indices=train_indices)

        return process_dataset(x_train, y_train, x_test, y_test, points_dimension, interface)

    @staticmethod
    def digits(n_train: int = 2000, n_test: int = 1000, points_dimension: int = 784,
               label1: int = 8, label2: int = 0,
               seed: int = None, interface: str = 'jax', path='mnist.npz'):
        root = get_root_path('Pulsed-Data-Reuploading-Quantum-Models')
        path = os.path.join(root, 'src/Sampler/mnist/mnist.npz')
        try:
            data = np_load(path)
        except:
            print("Archivo MNIST no encontrado. Descargando desde OpenML...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            X, y = mnist.data, mnist.target.astype(int)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np_savez_compressed(path, X=X, y=y)
            data = np_load(path)

        X, y = data['X'], data['y']
        mask = (y == int(label1)) | (y == int(label2))
        X_filtered, y_filtered = X[mask], y[mask]
        y_filtered = jnp.where(
            y_filtered == int(label1),
            0,
            1) if interface == 'jax' else qnp.where(
            y_filtered == label1,
            0,
            1)

        # Random subsets
        x_train, y_train, train_indices = get_random_subset(X_filtered, y_filtered, n_train, seed=seed)
        x_test, y_test, train_indices = get_random_subset(X_filtered, y_filtered, n_test, seed=seed,
                                                          exclude_indices=train_indices)

        dataset_processed = process_dataset(x_train, y_train, x_test, y_test, points_dimension, interface)
        return dataset_processed

    @staticmethod
    def iris(n_train: int = 25, n_test: int = 25, points_dimension: int = 4,
             label1: int = 1, label2: int = 2,
             seed: int = None, interface: str = 'jax') -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        0 → Iris setosa
        1 → Iris versicolor
        2 → Iris virginica
        """
        iris = load_iris()
        X, y = iris.data, iris.target
        mask = (y == label1) | (y == label2)
        X_filtered, y_filtered = X[mask], y[mask]
        y_filtered = jnp.where(
            y_filtered == label1,
            0,
            1) if interface == 'jax' else qnp.where(
            y_filtered == label1,
            0,
            1)

        # Adjust number of points
        if n_train + n_test > 100:
            n_train = 70
            n_test = 30
            warn("Number of training points adjusted to 70 and number of test points adjusted to 30.", stacklevel=2)
        if n_test > (len(y_filtered) - n_train):
            n_test = max(len(y_filtered) - n_train, 0) or 25
            warn("Number of test points adjusted to 50.", stacklevel=2)

        # Random subsets
        x_train, y_train, train_indices = get_random_subset(X_filtered, y_filtered, n_train, seed=seed)
        if n_test > 0:
            x_test, y_test, train_indices = get_random_subset(X_filtered, y_filtered, n_test, seed=seed,
                                                              exclude_indices=train_indices)
        else:
            x_test = None
            y_test = None

        return process_dataset(x_train, y_train, x_test, y_test, points_dimension, interface)

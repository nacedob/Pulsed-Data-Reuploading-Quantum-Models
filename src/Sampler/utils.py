from icecream import ic
import numpy as np
import jax
from jax import numpy as jnp
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from typing import Sequence


def generate_random_points(n_points, spread, point_size: int, interface: str = 'jax', seed: int = None):
    if interface == 'jax':
        if seed is None:
            seed = np.random.randint(0, 10000)
        key = jax.random.PRNGKey(seed)
        return spread * (2 * jax.random.uniform(key, shape=(n_points, point_size)) - 1)
    elif interface == 'pennylane':
        return spread * (2 * np.random.rand(n_points, point_size) - 1)



def get_random_subset(x_data, y_data, n_samples, seed=None, exclude_indices=None):
    """
    Toma un conjunto de datos y etiquetas, y devuelve un subconjunto aleatorio de tamaño n_samples,
    excluyendo índices en exclude_indices.

    Args:
    - x_data (list): Lista de características.
    - y_data (list): Lista de etiquetas.
    - n_samples (int): Número de muestras deseadas.
    - exclude_indices (set or list): Índices a excluir del muestreo.

    Returns:
    - (list, list, set): Subconjunto de características, etiquetas y los índices usados.
    """
    total_indices = set(range(len(x_data)))
    if exclude_indices is None:
        exclude_indices = set()
    else:
        exclude_indices = set(exclude_indices)

    available_indices = list(total_indices - exclude_indices)

    if n_samples > len(available_indices):
        raise ValueError(f"El tamaño solicitado ({n_samples}) excede el número de muestras disponibles ({len(available_indices)}).")

    if seed is not None:
        random.seed(seed)

    subset_indices = random.sample(available_indices, n_samples)

    x_subset = [x_data[i] for i in subset_indices]
    y_subset = [y_data[i] for i in subset_indices]

    return x_subset, y_subset, set(subset_indices)


def reduce_dimension(data, new_dim: int = 4, feature_range: tuple = None) -> Sequence:
    """
    Reduce the dimensionality of the input data using PCA and optionally scale the features.

    This function applies a pipeline of data preprocessing steps:
    1. Standardization of the input data.
    2. Principal Component Analysis (PCA) for dimensionality reduction.
    3. Optional scaling of the reduced data to a specified feature range.

    Parameters:
    data (array-like): The input data to be reduced in dimensionality.
    new_dim (int, optional): The number of dimensions to reduce the data to. Defaults to 4.
    feature_range (tuple, optional): The desired range of transformed data.
                                     If provided, the data will be scaled to this range.
                                     Defaults to None (no scaling).

    Returns:
    numpy.ndarray: The transformed data with reduced dimensionality and optional scaling.
    """
    steps = [
        ('normalize', StandardScaler()),
        ('pca', PCA(n_components=new_dim, svd_solver='full')),
    ]
    if feature_range is not None:
        steps += [('scaler', MinMaxScaler(feature_range=feature_range))]  # for instance (-1, 1)
    pipeline = Pipeline(steps)
    data_reduced = pipeline.fit_transform(data)
    return data_reduced

def scale_points(dataset: np.ndarray, scale_range: tuple = None, center: bool=False) -> np.ndarray:
    """
    Given a dataset, it scales the points so they are in the scale_range (-1, 1). This is done to create
    a meaningful encoder -> Rot(pi * x)
    :param dataset:
    :return: scaled dataset
    """
    if scale_range is None:
        scale_range = (-1, 1)
    scaler = MinMaxScaler(feature_range=scale_range)
    scaled_data = scaler.fit_transform(dataset)
    if center:
        scaler = StandardScaler(with_mean=True, with_std=False)
        scaled_data = scaler.fit_transform(scaled_data)

    return scaled_data

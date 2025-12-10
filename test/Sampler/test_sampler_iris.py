import jax.numpy as jnp
import numpy as np
from src.Sampler import MNISTSampler

def test_iris_function():
    n_train, n_test = 80, 40
    points_dimension = 3
    label1, label2 = 0, 2
    seed = 42

    x_train, y_train, x_test, y_test = MNISTSampler.iris(n_train, n_test, points_dimension, label1, label2, seed, interface='jax')

    # Check types
    assert isinstance(x_train, jnp.ndarray), "x_train debe ser jnp.ndarray"
    assert isinstance(y_train, jnp.ndarray), "y_train debe ser jnp.ndarray"
    assert isinstance(x_test, jnp.ndarray), "x_test debe ser jnp.ndarray"
    assert isinstance(y_test, jnp.ndarray), "y_test debe ser jnp.ndarray"

    # Check shapes
    assert x_train.shape == (n_train, points_dimension), f"x_train shape incorrecta: {x_train.shape}"
    assert y_train.shape == (n_train,), f"y_train shape incorrecta: {y_train.shape}"
    assert x_test.shape == (n_test, points_dimension), f"x_test shape incorrecta: {x_test.shape}"
    assert y_test.shape == (n_test,), f"y_test shape incorrecta: {y_test.shape}"

    # Check labels are 0 or 1
    assert jnp.all((y_train == 0) | (y_train == 1)), "y_train tiene etiquetas distintas de 0 o 1"
    assert jnp.all((y_test == 0) | (y_test == 1)), "y_test tiene etiquetas distintas de 0 o 1"

    print("Test iris passed.")

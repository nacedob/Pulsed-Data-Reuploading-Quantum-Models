from typing import Callable
from jax import numpy as jnp


def gaussian(amplitude: float, sigma: float, duration: float, *args, **kwargs) -> Callable:
    t_center = duration / 2
    return lambda t: amplitude * jnp.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))

def constant(amplitude, *args, **kwargs) -> Callable:
    return lambda t: amplitude


def sin(amplitude, freq, phase, *args, **kwargs) -> Callable:
    return lambda t: amplitude * jnp.sin(freq * t + phase)

"""
This script shows how pulses can be defined in the PulsedQNN.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Tuple
from src.Sampler import Sampler3D
from src.QNN.PulsedQNN import PulsedQNN
from warnings import filterwarnings
from jax import numpy as jnp

filterwarnings("ignore", category=RuntimeWarning)
filterwarnings("ignore", category=FutureWarning)

seed = 42
num_qubits = 2
num_layers = 4

# ---------------------------------------------------------------------------------------------------------------------
# Define an example dataset
# ---------------------------------------------------------------------------------------------------------------------


def get_dataset() -> Tuple[jnp.ndarray, jnp.ndarray]:
    return Sampler3D.spiral(n_points=100, seed=seed)


# ---------------------------------------------------------------------------------------------------------------------
# Gaussian envelope with different parameters for each qubit
# ---------------------------------------------------------------------------------------------------------------------
def gaussian_qnn() -> PulsedQNN:
    return PulsedQNN(
        num_qubits=num_qubits,
        num_layers=num_layers,
        noise=False,
        seed=seed,
        # Pulse parameters
        pulse_shape='gaussian',
        pulse_params=[{'sigma': 5.0}, {'sigma': 7.0}],  # 2 qubits, different parameters
        # Gate durations
        duration_1q_pulse=10,
        duration_2q_pulse=100,
        # Dataset interface
        interface='jax'   # 'numpy'
    )


# ---------------------------------------------------------------------------------------------------------------------
# Sine envelope with commmon parameters for each qubit
# ---------------------------------------------------------------------------------------------------------------------
def sine_qnn() -> PulsedQNN:
    return PulsedQNN(
        num_qubits=num_qubits,
        num_layers=num_layers,
        noise=False,
        seed=seed,
        # Pulse parameters
        pulse_shape='gaussian',
        pulse_params={'freq': 5.0, 'phase': 0.0},  # 2 qubits, common parameters
        # Gate durations
        duration_1q_pulse=10,
        duration_2q_pulse=100,
        # Dataset interface
        interface='jax'   # 'numpy'
    )


# ---------------------------------------------------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Get dataset
    X_train, y_train = get_dataset()

    # Gaussian pulse performance
    print("\nGaussian QNN performance:")
    qnn_gaussian = gaussian_qnn()
    acc = qnn_gaussian.get_accuracy(X_train, y_train)
    print(f"Accuracy: {acc:.4f}")
    
    # Sine pulse performance
    print("Sine QNN performance:")
    qnn_sine = sine_qnn()
    acc = qnn_sine.get_accuracy(X_train, y_train)
    print(f"Accuracy: {acc:.4f}")

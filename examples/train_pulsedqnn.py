"""
This script is an example of how to train a simple Pulsed QNN
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sklearn.datasets import make_classification
from jax import numpy as jnp
from warnings import filterwarnings
from src.QNN.PulsedQNN import PulsedQNN
from typing import Tuple


filterwarnings("ignore", category=RuntimeWarning)
filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------------------------------------------------
# 0. Experiment parameters
# ---------------------------------------------------------------------------------------------------------------------
seed = 42
# QNN Architecture parameters
n_qubits = 1
n_layers = 4
# Pulse parameters
pulse_shape = 'gaussian'
duration_1q_pulse = 10
duration_2q_pulse = 100
# Train parameters
n_epochs = 30
optimizer = 'adam'
learning_rate = 0.01

# ---------------------------------------------------------------------------------------------------------------------
# 1. Define the QNN
# ---------------------------------------------------------------------------------------------------------------------


def get_qnn() -> PulsedQNN:
    return PulsedQNN(
        num_qubits=n_qubits,
        num_layers=n_layers,
        noise=False,
        seed=seed,
        # Pulse parameters
        pulse_shape=pulse_shape,
        duration_1q_pulse=duration_1q_pulse,
        duration_2q_pulse=duration_2q_pulse,
        # Dataset interface
        interface='jax'   # 'numpy'
    )

# ---------------------------------------------------------------------------------------------------------------------
# 2. Load dataset
# It is important to know that jax module is used for optimization, so datasets can must be converted to jax.numpy
# arrays. Regular np arrays can be used if the constructor is used with interface = 'numpy' in the GateQNN, but 
# currently it is not supported for the PulsedQNN.
# ---------------------------------------------------------------------------------------------------------------------


def make_classification_dataset_jax(
    n_features: int,
    n_train: int,
    n_test: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    n_total = n_train + n_test

    X, y = make_classification(
        n_samples=n_total,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=2,
        random_state=seed,
    )

    # Convert to jax.numpy
    X = jnp.array(X)
    y = jnp.array(y)

    # Split
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------------------------------------------------
# 3. Train the QNN
# ---------------------------------------------------------------------------------------------------------------------
def train(
    qnn: PulsedQNN,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray
) -> None:
    qnn.train(
        data_points_train=X_train,
        data_labels_train=y_train,
        data_points_test=X_test, 
        data_labels_test=y_test, 
        n_epochs=n_epochs,
        optimizer=optimizer,
        optimizer_parameters={'lr': learning_rate},
        silent=False
    )
    return None


# ---------------------------------------------------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------------------------------------------------
def main():
    # 1. Define the QNN
    qnn = get_qnn()
    # 2. Load dataset
    X_train, y_train, X_test, y_test = make_classification_dataset_jax(
        n_features=2,
        n_train=100,
        n_test=50,
    )
    # 3. Train the QNN
    print('Training the QNN...')
    train(qnn, X_train, y_train, X_test, y_test)
    
    # 4. Print final accuracies
    print('Final accuracies:')
    print(f"Train accuracy: {qnn.get_accuracy(X_train, y_train):.4f}")
    print(f"Test accuracy: {qnn.get_accuracy(X_test, y_test):.4f}")
    
if __name__ == "__main__":
    main()
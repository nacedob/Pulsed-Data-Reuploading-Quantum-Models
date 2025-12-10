from icecream import ic
from src.Pennypulse.src.pennypulse import pulse1q
from src.Pennypulse.src.pennypulse import shapes
from src.visualization import plot_bloch_points
import pennylane as qml
import unittest
from jax import numpy as np


class Test1QPulse(unittest.TestCase):

    def test_rotation(self):
        qubit_freq = 1
        drive_freq = qubit_freq
        amplitude = 20
        amplitude_function = shapes.constant(amplitude)

        @qml.qnode(qml.device('default.qubit', wires=1))
        def circuit(duration, phase):
            pulse1q(qubit_freq, drive_freq, phase, amplitude_function, duration, wire=0)
            return qml.state()

        results_phase0 = []
        for t in np.linspace(0.1, 50, 20):
            results_phase0.append(circuit(duration=t, phase=0))
        results_phasepi = []
        for t in np.linspace(0.1, 50, 20):
            results_phasepi.append(circuit(duration=t, phase=np.pi/2))
        ax = plot_bloch_points(results_phase0, point_color='red', show=False)
        plot_bloch_points(results_phasepi, point_color='blue', ax=ax, show=True)
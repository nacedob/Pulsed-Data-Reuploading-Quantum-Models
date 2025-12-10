from typing import Sequence, Optional, List
import pennylane as qml
from pennylane.typing import TensorLike
import jax
from math import pi
from .BaseQNN import BaseQNN, cartesian_to_spherical
from .constants import DEFAULT_1Q_PULSE_DURATION, DEFAULT_2Q_PULSE_DURATION

jax.config.update("jax_enable_x64", True)


class GateQNN(BaseQNN):

    def __init__(
            self,
            num_qubits: int = 2,
            num_layers: int = 5,
            regularization: float = 0,
            n_workers: int = 1,
            encoding: str = "original",
            interface: str = "jax",
            debug_noise: bool = False,
            realistic_gates: bool = True,
            duration_1q_pulse: float = DEFAULT_1Q_PULSE_DURATION,
            duration_2q_pulse: float = DEFAULT_2Q_PULSE_DURATION,
            seed: Optional[int] = None,
            noise: bool = False,
            noise_sources: Optional[List[str]] = None,
            noise_parameters: Optional[dict] = None
    ):
        """
        Quantum Neural Network (QNN) with a gate-based architecture.

        Args:
            num_qubits: The number of qubits in the QNN. Defaults to 2.
            num_layers: The number of layers in the QNN. Defaults to 5.
            n_workers: The number of workers for parallel execution (currently not effective). Defaults to 1.
            interface: The interface used for the QNN ('jax' or 'pennylane'). Defaults to 'jax'.
            realistic_gates: Whether to use realistic gates. Defaults to True.
            seed: The random seed used for reproducibility. Defaults to None.
            noise: Whether to apply noise to the circuit. Defaults to False.

        Attributes:
            num_qubits: The number of qubits in the QNN.
            num_layers: The number of layers in the QNN.
            params_per_layer: The number of parameters per layer = 3.
            interface: The interface used for the QNN ('jax' or 'pennylane').
            seed: The random seed used for reproducibility.
            params: The parameters of the QNN circuit.
            projection_angles: The angles used for projection measurements.
            trained: A boolean indicating whether the QNN has been trained.
            training_info: A dictionary to store training information.
            dev: The PennyLane quantum device used for simulations.
            n_workers: The number of workers for parallel execution (currently not effective).
            name: The name of the specific QNN architecture.
            model_name: The name of the overall model.
            realistic_gates: A boolean indicating whether to use realistic
                         (hardware-efficient) gates or arbitrary rotations.

        Raises:
            ValueError: If the number of qubits is greater than the number of layers + 1.
        """
        super().__init__(
            num_qubits,
            num_layers,
            n_workers,
            regularization=regularization,
            params_per_layer=3,
            interface=interface,
            seed=seed,
            noise=noise,
            duration_1q_pulse=duration_1q_pulse,
            duration_2q_pulse=duration_2q_pulse,
            noise_sources=noise_sources,
            noise_parameters=noise_parameters
        )

        assert encoding in [
            "original",
            "spherical",
        ], f"Unknown encoding {encoding}. Choose between original and spherical"
        self.encoding = encoding
        self.name = f"GateQNN with {num_qubits=}, {num_layers=}. Interface={interface}. N_workers={n_workers}"
        self.model_name = "GateQNN" if self.encoding == "original" else "GateQNN_spherical"
        self.realistic_gates = realistic_gates
        self.debug_noise = debug_noise
        if self.debug_noise:
            assert self.noise

    def _base_circuit(self, x, params=None) -> None:
        """
        This is the base circuit of the gate model.
        In few words
            - as encoding gate it uses an arbitrary SU(2) rotation RZ(x[2])RY(x[1])RZ(x[0]).
            - as parametrized one qubit gate uses an arbitrary SU(2) rotation RZ(param[2])RY(param[1])RZ(param[0]).
            - as parametrized two qubit gate:
                · If realistic_gates: multirotations RXZ(0, wire)
                · If not realistic_gates: controlled arbitrary SU(2) rotation RZ(param[2])RY(param[1])RZ(param[0]).
                    Control wire is the corresponding qubit and target is 0.
        Args:
            x: (array-like) single input point
            params: (array-like) parameters for the circuit
        Returns:
            None
        """
        if params is None:
            params = self.params

        n_qubits = (params.shape[0] + 1) // 2
        for layer in range(self.num_layers):
            if self.debug_noise and layer > 19:
                for qubit in range(n_qubits):
                    # one per encoding and one per parametric gates
                    self._apply_one_qubit_noise(qubit=qubit)
                    self._apply_one_qubit_noise(qubit=qubit)
                    if qubit != 0:
                        self._apply_two_qubit_noise(qubits=[0, qubit])
            else:
                for qubit in range(n_qubits):
                    self._encoding_function(x, qubit)
                    if qubit != 0:
                        self._one_qubit_gate(params[2 * qubit - 1][layer, :], wires=qubit)
                        self._two_qubit_gate(params[2 * qubit][layer, :], wires=[0, qubit])
                    else:
                        self._one_qubit_gate(params[qubit][layer, :], wires=qubit)

    def _encoding_function(self, x: Sequence[TensorLike], qubit: int) -> None:
        if self.encoding == "original":
            qml.Rot(*(x * pi), wires=qubit)  # TODO mirar que este parentesis no afecte
        else:
            spherical_coords = cartesian_to_spherical(x)
            qml.Rot(*spherical_coords, wires=qubit)

    def _one_qubit_gate(self, params: Sequence[TensorLike], wires: int) -> None:
        if self.noise:
            qml.RZ(params[0], wires=wires)
            qml.RY(params[1], wires=wires)
            self._apply_one_qubit_noise(qubit=wires)
            qml.RZ(params[2], wires=wires)
        else:
            qml.Rot(*params, wires=wires)

    def _two_qubit_gate(
            self, params: Sequence[TensorLike], wires: Sequence[int]
    ) -> None:
        if self.realistic_gates:
            if self.noise:
                raise NotImplementedError(
                    "Realistic gates with noise not implemented yet"
                )
            # RXZ rotation
            qml.Hadamard(wires=wires[0])
            qml.MultiRZ(theta=params[0], wires=wires)  # params 2 and 3 are discarded
            qml.Hadamard(wires=wires[0])

        else:
            # CRot gate
            if self.noise:
                self._noisy_crot_transpiled(params, wires=wires)
            else:
                qml.CRot(*params, wires=wires)

    def _noisy_crot_transpiled(
            self, params: Sequence[TensorLike], wires: Sequence[int]
    ) -> None:
        """
        CRot circuit (C_(RZRYRZ) transpiled into the natural gates
        https://quantumcomputing.stackexchange.com/questions/37633/two-level-unitary-gates-are-decomposed-into-single-qubit-gates-and-cnot-gates
        Args:
            params: three euler angles
            params: two length tuple (control, target)
        """
        # A
        qml.RZ(params[0], wires=wires[1])
        qml.RY(params[1] / 2, wires=wires[1])
        self._apply_one_qubit_noise(wires[1])
        # CNOT
        self._noisy_cnot_transpiled(wires)  # this includes noise
        # B
        qml.RY(-params[1] / 2, wires=wires[1])
        self._apply_one_qubit_noise(wires[1])
        qml.RZ(-(params[0] + params[2]) / 2, wires=wires[1])
        # CNOT
        self._noisy_cnot_transpiled(wires)  # this includes noise
        # C
        qml.RZ(-(params[0] - params[2]) / 2, wires=wires[1])

    def _noisy_cnot_transpiled(self, wires: Sequence[int]) -> None:
        """
        # CNOT transpiled with noise
               -----------   -----------
        --*----| RZ(pi/2)|---|CR(-pi/2)|---
          |    -----------   -----------
          |     -----------     |
        --X----| RX(pi/2) |-----*------
                -----------
        """
        # RX on control qubit (wires[0])
        qml.RX(pi / 2, wires=wires[0])
        self._apply_one_qubit_noise(wires[0])
        # RZ on target qubit (wires[1])
        qml.RZ(pi / 2, wires=wires[1])
        # CR (control is wires[1], target is wires[0]) (recall CR[control, target] is R[X_control Z_target])
        qml.Hadamard(wires=wires[1])
        qml.MultiRZ(theta=-pi / 2, wires=wires[::-1])
        qml.Hadamard(wires=wires[1])
        self._apply_two_qubit_noise(wires)

    def _define_gate_durations(self, duration_1q_pulse, duration_2q_pulse):
        return [duration_1q_pulse, duration_2q_pulse]

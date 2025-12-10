import pennylane as qml
import jax
from .BaseQNN import BaseQNN
from jax import numpy as jnp
from warnings import warn
from .constants import fakemanila2q, brisbane, DEFAULT_1Q_PULSE_DURATION, DEFAULT_2Q_PULSE_DURATION
from typing import Optional, Union
import src.pennypulse as pennypulse

jax.config.update("jax_enable_x64", True)

AVAILABLE_SHAPES = ['gaussian', 'constant', 'sin']
default_params_gaussian = {'sigma': 1, 'duration': 'TBD'}  # if not provided duration, TBD will produce an error



class PulsedQNN(BaseQNN):

    def __init__(self,
                 num_qubits: int = 2,
                 num_layers: int = 5,
                 n_workers: int = 1,
                 regularization: float = 0,
                 interface: str = 'jax',
                 pulse_shape: Union[list[str], str] = 'constant',
                 pulse_params: Optional[list[dict]] = None,
                 n_trotter: int = 5,
                 backend: str = 'brisbane',
                 encoding: str = 'pulsed',
                 duration_1q_pulse: float = DEFAULT_1Q_PULSE_DURATION,
                 duration_2q_pulse: float = DEFAULT_2Q_PULSE_DURATION,
                 constant4amplitude: float = 1,
                 seed: Optional[int] = None,
                 noise: bool = False,
                 noise_sources: Optional[list[str]] = None,
                 noise_parameters: Optional[dict] = None
                 ):
        """
        Quantum Neural Network (QNN) with a pulsed-based architecture (parametrized pulses).
        Inside this class, there are two possible models to define: the (pure) pulsed model and a mixed strategy.
        They differ from the encoding strategy. The pulse model encodes directly with a pulse base strategy (based in
        spherical coordinateS) while the mixed model shares the encoding strategy with the gate (GateQNN) model:
        an arbitrary SU(2) rotation: RZ(x[2])RY(x[1])RZ(x[0])

        Args:
            num_qubits: The number of qubits in the QNN. Defaults to 2.
            num_layers: The number of layers in the QNN. Defaults to 5.
            n_workers: The number of workers for parallel execution (currently not effective). Defaults to 1.
            interface: The interface used for the QNN ('jax' only). Defaults to 'jax'.
            pulse_shape: A string or list of strings specifying the pulse shape(s). Defaults to 'constant'.
            pulse_params: A list of dictionaries containing pulse shape parameters. Defaults to None.
            n_trotter: The number of Trotter steps. Defaults to 5.
            backend: The backend to simulate. Defaults to 'brisbane'.
            encoding: The encoding method. Defaults to 'pulsed'.
            duration_1q_pulse: Duration of single-qubit gate in ns. Defaults to 300.
            duration_2q_pulse: Duration of two-qubit gate in ns. Defaults to 660.
            constant4amplitude: Constant factor for pulse amplitudes. Defaults to 1.
            seed: The random seed used for reproducibility. Defaults to None.
            noise: A boolean indicating whether to apply noise to the circuit. Defaults to False.

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
            pulse_shapes: A list of strings specifying the shape of the pulses for each qubit.
            pulse_params: A list of dictionaries containing parameters for the pulse shapes.
            n_trotter: The number of Trotter steps used for Hamiltonian simulation.
            backend: The name of the backend to simulate.
            encoding: The type of encoding used for the input data.
            duration_1q_pulse: The duration of a single-qubit gate in nanoseconds.
            duration_2q_pulse: The duration of a two-qubit gate in nanoseconds.
            constant4amplitude: A constant factor to multiply parameterized pulse amplitudes.
            q_freq: A list of qubit frequencies for the chosen backend.
            connections: A list of qubit connections for the chosen backend.
            coupling: The coupling strength between connected qubits.
            AMPLITUDE_BOUNDARY: The lower and upper bounds for pulse amplitudes.

        Raises:
            ValueError:
              - If the interface is not 'jax'.
              - If the length of `pulse_shape` does not match the number of qubits.
              - If an invalid pulse shape is provided.
              - If an invalid backend is provided.
              - If num_qubits is greater than two (TODO: generalize model)
        """

        if interface != 'jax':
            raise ValueError('Only JAX interface is currently supported.')

        # Gate durations and shape
        self.duration_1q_pulse, self.duration_2q_pulse = self._define_gate_durations(duration_1q_pulse, duration_2q_pulse)

        super().__init__(num_qubits, num_layers, n_workers, params_per_layer=4, interface=interface, seed=seed,
                         noise=noise, noise_parameters=noise_parameters, noise_sources=noise_sources,
                         regularization=regularization)

        # Constant to multiply parametrized pulse amplitudes
        self.constant4amplitude = constant4amplitude

        # Get gate pulse shapes and process
        if isinstance(pulse_shape, str):
            # If a single pulse shape is provided, apply it to all qubits
            self.pulse_shapes = [pulse_shape] * self.num_qubits
        else:
            # If multiple pulse shapes are provided, apply them to corresponding qubits
            if len(pulse_shape) != self.num_qubits:
                raise ValueError(f'Pulse shape list length must match the number of qubits ({self.num_qubits}).')
            self.pulse_shapes = pulse_shape

        if not all(shape in AVAILABLE_SHAPES for shape in self.pulse_shapes):
            raise ValueError(f'Invalid pulse shape(s): {pulse_shape}. Supported shapes are {AVAILABLE_SHAPES}.')

        self.amplitude_func_list = [getattr(pennypulse.shapes, shape) for shape in self.pulse_shapes]

        # Handle pulse parameters
        if pulse_params is None:
            pulse_params = [{} for _ in range(self.num_qubits)]  # Default to empty dicts for each qubit
        elif isinstance(pulse_params, dict):
            warn('Pulse parameters introduced is a dict. It will be used for all qubits.')
            pulse_params = [pulse_params for _ in range(self.num_qubits)]  # Apply same params to all qubits
        elif isinstance(pulse_params, list):
            if not all(isinstance(el, dict) for el in pulse_params):
                raise ValueError('Pulse parameters introduced is not a list of dicts.')
            if len(pulse_params) != self.num_qubits:
                raise ValueError(
                    f'Pulse parameters list must have the same length as the number of qubits ({self.num_qubits}).')
        self.pulse_params = pulse_params

        for i in range(self.num_qubits):
            if self.pulse_shapes[i] == 'gaussian':
                for key, value in default_params_gaussian.items():
                    if key not in self.pulse_params[i]:
                        self.pulse_params[i][key] = value

        # Get backend
        if 'manila' in backend.lower():
            back = fakemanila2q
        elif backend.lower() == 'brisbane':
            back = brisbane
        else:
            raise ValueError(f'Invalid backend: {backend}. Supported backends are "fakemanila" and "brisbane".')

        self.q_freq: list = list(back['freq'].values())
        if self.num_qubits == 2:
            self.connections = [[0, 1]]
            self.coupling: float = back['coupling']['q0q1']
        elif self.num_qubits == 1:
            self.connections = None
            self.coupling = None
        else:
            raise ValueError(f'Invalid number of qubits: {self.num_qubits}. Supported qubit numbers are 1 and 2.')

        self.n_trotter = n_trotter

        # Store accumulated phases (TODO)
        # self.pulse_phases = jnp.zeros(2) if self.interface == 'jax' else qml.numpy.zeros(2, requires_grad=True)

        # Physical pulse parameters limits
        self.AMPLITUDE_BOUNDARY = [1e-3, 10000]

        self.name = f'PulsedQNN with {num_qubits=}, {num_layers=} from {backend=} and {encoding=}.'
        self.model_name = f'PulsedQNN_encoding_{encoding}'


    def _encoding(self, x: jnp.array, wire: int) -> None:
        """
        Encodes the classical data into the quantum circuit as in the gate model: using an Arbitrary SU(2) rotation.
        RZ(x[2])RY(x[1])RZ(x[0])
        Args:
            x (array-like): classical single point to be encoded
            wire (int): qubit in which encode the data
        Returns:
            None
        """
        if self.noise:
            qml.RZ(x[0], wires=wire)
            qml.RY(x[1], wires=wire)
            self._apply_one_qubit_noise(wire)
            qml.RZ(x[2], wires=wire)
        else:
            qml.Rot(*(jnp.pi * x), wires=wire)

    # Core circuit
    def _base_circuit(self, x, params) -> None:
        """
        This is the base circuit of the pulsed/mixed model.
        In few words
            - as encoding gate it the encoding_operation (defined above).
            - as parametrized one qubit gate uses VZ(param[0]) Pulse(param[1:2]) VZ(param[3]). This single pulse varies
                the amplitude (rotation angle) and phase (rotation axis)
            - as parametrized two qubit gate. it uses a single pulse whose parameters are the amplitude (rotation angle),
                and the frequency detuning with respect to the other qubit frequency and phase (rotation axis). Then
                it applies a VZ rotation.
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
            for qubit in range(n_qubits):
                # X encoding into 1 qubit gate
                self._encoding(x, wire=qubit)   # this includes noise
                # Actuation of params
                if qubit != 0:
                    self._create_1q_gate(params=params[2 * qubit - 1][layer, :], wire=qubit)    # this includes noise
                    self._create_2q_gate(params=params[2 * qubit][layer, :], wires=[0, qubit])  # this includes noise
                else:
                    self._create_1q_gate(params=params[qubit][layer, :], wire=qubit)            # this includes noise

    # Pulsed gate design
    def _create_1q_gate(self, params: list, wire: int) -> None:
        """
        As in GateQNN model, I perform:
         RZ(params[4]), RY(amplitude = params[1], freq_shift = params[2]) RZ(params[0])
        Args:
            params (list): params vec of length 4 (Exactly)
            wire (int): wire to apply to the gate
        Returns:
             None
        """
        if len(params) != 4:
            raise ValueError("Params must be a list of length 4.")

        # Virtual Z rotation
        qml.RZ(params[0], wires=wire)

        # Arbitrary pulse rotation in the XY plane (amplitude = params[1], freq_shift = params[2])
        pulse_params = self.pulse_params.copy()
        pulse_params[wire]['duration'] = self.duration_1q_pulse
        amplitude_func = lambda t: self.amplitude_func_list[wire](amplitude=params[1] * self.constant4amplitude,
                                                                  **pulse_params[wire])(t)   #  * 20
        pennypulse.pulse1q(q_freq=self.q_freq[wire],
                           amplitude_func=amplitude_func,
                           drive_freq=self.q_freq[wire],
                           drive_phase=params[2],   # + self.pulse_phases[wire],
                           duration=self.duration_1q_pulse,
                           n_trotter=self.n_trotter,
                           wire=wire,
                           )

        if self.noise:
            self._apply_one_qubit_noise(wire)

        # Virtual Z rotation
        qml.RZ(params[3], wires=wire)

    def _create_2q_gate(self, params: list, wires: list[int]) -> None:
        """
        I perform a single pulse to create entanglement. The reference pulse corresponds to the CR gate, but parameters
        can deviate it from the exact CR parameters if convenient.
        The pulse is applied to wire[0] (always 0) with qubit freq wire[1] + params[0].
        The amplitude of this pulse is params[1] and its phase params[2].
        Finally, a VZ rotation is applied to qubit 0 with rotation angle params[3]
        Args:
            params (list): params vec of length 4 (Exactly) -> (freq_dephase, amplitude, phase, VZ rot angle)
            wires (list): wires to apply the CR gate. The first one is the qubit experimenting the pulse, and the
                            second one is the qubit whose frequency is chosen as the pulse frequency reference.
        Returns:
            None
        """
        if len(params) != 4:
            raise ValueError("Params must be a list of length 4.")

        amplitude_func = lambda t: self.amplitude_func_list[wires[0]](amplitude=params[1] * self.constant4amplitude,
                                                                      **self.pulse_params[wires[0]])(t)  #  * 20
        pennypulse.transmon_trotter_suzuki_2q_drive1q(q_freqs=self.q_freq,
                                                      coupling=self.coupling,
                                                      amplitude_func=amplitude_func,
                                                      drive_freq=self.q_freq[wires[1]] + params[0],
                                                      drive_phase=params[2],
                                                      wire=wires[0],
                                                      n_trotter=self.n_trotter,
                                                      duration=self.duration_2q_pulse)
        if self.noise:
            self._apply_two_qubit_noise(wires)
        qml.RZ(params[3], wires=wires[1])      # use the other parameter

    #
    # Utils
    #
    def _clip_params(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Clips the parameters to stay inside the validity parameters. Only constrained parameter is amplitude.
        It is called after each epoch in the training function from parent class BaseQNN

        Args
            p: (array-like) parameters
        Returns:
            (array-like) parameters after clipping
        """
        p = p.at[:, :, 1].set(jnp.clip(p[:, :, 1], *self.AMPLITUDE_BOUNDARY))
        return p


    def _define_gate_durations(self, duration_1q_pulse: float, duration_2q_pulse: float) -> [float, float]:
        """
        Both cases use just one pulse, so it is directly the pulse duration
        """
        return [duration_1q_pulse, duration_2q_pulse]
    
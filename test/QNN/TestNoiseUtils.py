import pennylane as qml
import numpy as np
from icecream import ic

device = qml.device('default.mixed', wires=1)
random_angle = np.random.rand(2)
p0 = np.random.rand()
p1 = np.random.rand()
gamma_phase = np.random.rand()
gamma_amplitude = np.random.rand()


I = qml.I(1).matrix()
X = qml.X(1).matrix()
Y = qml.Y(1).matrix()
Z = qml.Z(1).matrix()

tolerance = 1e-5


def get_krauss_matrices_depolarizing(p):
    k0 = np.sqrt(1 - p) * I
    k1 = np.sqrt(p / 3) * X
    k2 = np.sqrt(p / 3) * Y
    k3 = np.sqrt(p / 3) * Z
    return [k0, k1, k2, k3]

def get_krauss_matrices_phase_damping(g):
    k0 = np.array([[1, 0], [0, np.sqrt(1 - g)]])
    k1 = np.array([[0, 0], [0, np.sqrt(g)]])
    return [k0, k1]


def get_krauss_matrices_amplitude_damping(g):
    k0 = np.array([[1, 0], [0, np.sqrt(1 - g)]])
    k1 = np.array([[0, np.sqrt(g)], [0, 0]])
    return [k0, k1]


def base_circuit():
    qml.RZ(random_angle[0], wires=0)
    qml.RX(random_angle[1], wires=0)


def test_depolarizing_channel():

    @qml.qnode(device)
    def qml_circuit():
        base_circuit()
        qml.DepolarizingChannel(p0, wires=0)
        qml.DepolarizingChannel(p1, wires=0)
        return qml.expval(qml.PauliZ(0))


    @qml.qnode(device)
    def manual_circuit():
        base_circuit()
        qml.QubitChannel(get_krauss_matrices_depolarizing(p0), wires=0)
        qml.QubitChannel(get_krauss_matrices_depolarizing(p1), wires=0)
        return qml.expval(qml.PauliZ(0))

    qml_result = qml_circuit()
    manual_result = manual_circuit()
    ic(qml_result, manual_result)
    assert abs(qml_result - manual_result) < tolerance


def test_combinig_depolarizing_channel():
    krauss0 = get_krauss_matrices_depolarizing(p0)
    krauss1 = get_krauss_matrices_depolarizing(p1)

    combined_krauss = []
    for mat0 in krauss0:
        for mat1 in krauss1:
            combined_krauss.append(
                mat1 @ mat0      # TODO: da igual cambiar este orden wtf!?
            )

    @qml.qnode(device)
    def individual_circuit():
        base_circuit()
        qml.DepolarizingChannel(p0, wires=0)
        qml.DepolarizingChannel(p1, wires=0)
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(device)
    def combined_circuit():
        base_circuit()
        qml.QubitChannel(combined_krauss, wires=0)
        return qml.expval(qml.PauliZ(0))

    individual_result = individual_circuit()
    combined_result = combined_circuit()
    ic(individual_result, combined_result)
    assert abs(individual_result - combined_result) < tolerance

def test_combine_different_channels():

    @qml.qnode(device)
    def indivual_circuit():
        base_circuit()
        qml.DepolarizingChannel(p0, wires=0)
        qml.AmplitudeDamping(gamma_amplitude, wires=0)
        qml.PhaseDamping(gamma_phase, wires=0)
        return qml.expval(qml.PauliZ(0))

    combined_channel = []
    krauss0 = get_krauss_matrices_depolarizing(p0)
    krauss1 = get_krauss_matrices_amplitude_damping(gamma_amplitude)
    krauss2 = get_krauss_matrices_phase_damping(gamma_phase)

    for mat0 in krauss0:
        for mat1 in krauss1:
            for mat2 in krauss2:
                combined_channel.append(
                    mat2 @ mat1 @ mat0
                )

    @qml.qnode(device)
    def combined_circuit():
        base_circuit()
        qml.QubitChannel(combined_channel, wires=0)
        return qml.expval(qml.PauliZ(0))

    individual_result = indivual_circuit()
    combined_result = combined_circuit()
    ic(individual_result, combined_result)
    assert abs(individual_result - combined_result) < tolerance


def test_include_noise_combined():

    def noise(p, gamma_amplitude_, gamma_phase_):
        qml.DepolarizingChannel(p, wires=0)
        qml.AmplitudeDamping(gamma_amplitude_, wires=0)
        qml.PhaseDamping(gamma_phase_, wires=0)


    random_rot = np.random.rand(6) * np.pi * 2
    noise_args = np.array([1e-2, 0.1, 0.2])
    @qml.qnode(device)
    def decomposed_noise():
        qml.Rot(*random_rot[:3], wires=0)
        noise(*noise_args)
        qml.Rot(*random_rot[3:], wires=0)
        noise(*noise_args)
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(device)
    def combined_noise():
        qml.Rot(*random_rot[:3], wires=0)
        qml.Rot(*random_rot[3:], wires=0)
        noise(*(2 * noise_args))
        return qml.expval(qml.PauliZ(0))

    decomposed_result = decomposed_noise()
    combined_result = combined_noise()

    ic(decomposed_result, combined_result)
    assert abs(decomposed_result - combined_result) < tolerance * 1e3


if __name__ == '__main__':
    test_depolarizing_channel()
    test_combinig_depolarizing_channel()
    test_combine_different_channels()
    test_include_noise_combined()

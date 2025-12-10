import pennylane as qml
import unittest
import pandas as pd
from src.QNN import GateQNN
from src.Sampler import Sampler, MNISTSampler
from jax import numpy as np
from src.utils import  accuracy_score
import jax
from icecream import ic
from warnings import warn
from time import time
from src.utils import increase_dimensions


class GateQNNTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_set, cls.train_label = Sampler.circle(n_points=20)
        cls.test_set, cls.test_label = Sampler.circle(n_points=10)

        cls.n_qubits = 2
        cls.n_layers = 3
        cls.qnn = GateQNN(num_qubits=cls.n_qubits, num_layers=cls.n_layers)

    def test_cost_sequential(self):
        # Set up things
        key = jax.random.PRNGKey(0)  # Seed for reproducibility
        p = [
            jax.random.uniform(key, shape=(self.n_layers, 3))
            for _ in range(2 * self.n_qubits - 1)
        ]
        b = jax.random.uniform(key, shape=(1,))
        set = increase_dimensions(self.train_set, new_dimension=3)

        # Execute
        c = self.qnn.cost([p, b], set, self.train_label)
        self.assertGreaterEqual(c, 0)

    def test_cost_sequential_jit_first_time_Execution_slower(self):
        # Set up things
        key = jax.random.PRNGKey(0)  # Seed for reproducibility
        p = [
            jax.random.uniform(key, shape=(self.n_layers, 3))
            for _ in range(2 * self.n_qubits - 1)
        ]
        b = jax.random.uniform(key, shape=(1,))

        train_set, train_label = Sampler.circle(n_points=150)
        set = increase_dimensions(train_set, new_dimension=3)

        start_time = time()
        c_warm = self.qnn.cost([p, b], set, train_label).block_until_ready()
        time_warm_up = time() - start_time

        # Execute
        start_time = time()
        c_regular = self.qnn.cost([p, b], set, train_label).block_until_ready()
        time_regular = time() - start_time

        ic(time_warm_up, time_regular)
        self.assertEqual(c_warm, c_regular)
        self.assertGreater(time_warm_up, time_regular)

    def test_cost_vs_no_jitted_cost(self):
        key = jax.random.PRNGKey(0)  # Seed for reproducibility
        p = [
            jax.random.uniform(key, shape=(self.n_layers, 3))
            for _ in range(2 * self.n_qubits - 1)
        ]
        b = jax.random.uniform(key, shape=(1,))

        points, labels = Sampler.circle(n_points=500)
        set = increase_dimensions(points, new_dimension=3)
        points_warm_up, labels_warm_up = Sampler.circle(n_points=3)
        set_warm_up = increase_dimensions(points_warm_up, new_dimension=3)

        # jax.jit warm up
        self.qnn.cost([p, b], set_warm_up, labels_warm_up)

        start_time = time()
        c_jitted = self.qnn.cost([p, b], set, labels).block_until_ready()
        time_jitted = time() - start_time

        # Execute
        start_time = time()
        c_regular = self.qnn._cost_no_jitted([p, b], set, labels)
        time_regular = time() - start_time

        ic(time_jitted, time_regular)
        self.assertAlmostEqual(c_jitted, c_regular)
        self.assertGreater(time_regular, time_jitted)

    @unittest.skip('Parallel execution does not work as expected')
    def test_cost_parallel_working(self):
        # Set up things
        key = jax.random.PRNGKey(0)  # Seed for reproducibility
        p = [
            jax.random.uniform(key, shape=(self.n_layers, 3))
            for _ in range(2 * self.n_qubits - 1)
        ]
        b = jax.random.uniform(key, shape=(1,))

        set = increase_dimensions(self.train_set, new_dimension=3)

        # Execute
        qnn_parallel = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, n_workers=4)
        c = qnn_parallel.cost([p, b], set, self.train_label)
        self.assertGreaterEqual(c, 0)

    @unittest.skip('Parallel execution does not work as expected')
    def test_cost_parallel_faster(self):
        # Create bigger point set
        points, labels = Sampler.circle(n_points=2500)

        # Set up things
        key = jax.random.PRNGKey(0)  # Seed for reproducibility
        p = [
            jax.random.uniform(key, shape=(self.n_layers, 3))
            for _ in range(2 * self.n_qubits - 1)
        ]
        b = jax.random.uniform(key, shape=(1,))

        set = increase_dimensions(points, new_dimension=3)

        # Create QNN parallel
        qnn_parallel = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, n_workers=1)

        # Execute sequential
        start_time = time()
        c_seq = self.qnn.cost([p, b], set, labels).block_until_ready()
        time_seq = time() - start_time

        start_time = time()
        c_parallel = qnn_parallel.cost([p, b], set, labels).block_until_ready()
        time_parallel = time() - start_time

        # Assert same results
        ic(c_seq, c_parallel)
        self.assertAlmostEqual(c_seq, c_parallel, places=4)

        # Assert faster execution
        ic(time_seq, time_parallel)
        self.assertLess(time_parallel, time_seq)

    def test_cost_depends_on_parameters(self):
        set = increase_dimensions(self.train_set, new_dimension=3)

        key = jax.random.PRNGKey(0)  # Seed for reproducibility

        # First, check that cost changes when changing qnn parameters and keeping b fixed
        params_before = [
            jax.random.uniform(key, shape=(self.n_layers, 3))
            for _ in range(2 * self.n_qubits - 1)
        ]
        b_before = jax.random.uniform(key, shape=(1,))

        cost_before = self.qnn.cost([params_before, b_before], set, self.train_label)

        key = jax.random.PRNGKey(1)  # Change seed
        params_after = [
            jax.random.uniform(key, shape=(self.n_layers, 3))
            for _ in range(2 * self.n_qubits - 1)
        ]
        cost_after = self.qnn.cost([params_after, b_before], set, self.train_label)

        for p_before, p_after in zip(params_before, params_after):
            self.assertFalse((p_before == p_after).all())
        self.assertNotEquals(cost_after, cost_before)

        # Now keep qnn params fixed and vary bias
        b_after = jax.random.uniform(key, shape=(1,))
        cost_after_bias = self.qnn.cost([params_before, b_after], set, self.train_label)

        self.assertFalse((b_before == b_after).all())
        self.assertNotEquals(cost_after_bias, cost_before)

    def test_forward(self):
        result = self.qnn.forward(self.train_set)
        self.assertEqual(self.train_set.shape[0], result.shape[0])

    @unittest.skip('Parallel execution does not work as expected')
    def test_forward_parallel(self):
        # Create a bigger train set
        points, _ = Sampler.circle(n_points=200)

        # Run in parallel
        qnn_parallel = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, n_workers=4)
        start = time()
        result_parallel = qnn_parallel.forward(points).block_until_ready()
        t_parallel = time() - start

        # Run sequential
        start = time()
        result_sequential = self.qnn.forward(points).block_until_ready()
        t_sequential = time() - start

        # Test results and times
        ic(t_parallel, t_sequential)
        self.assertTrue(np.allclose(result_parallel, result_sequential))
        self.assertLess(t_parallel, t_sequential)

    def test_train_sequential(self):
        # run on simpler dataset
        train_set, train_label = Sampler.circle(n_points=20)
        test_set, test_label = Sampler.circle(n_points=10)

        # save init params
        init_params = self.qnn.params
        init_projection_params = self.qnn.projection_angles

        acc_before = self.qnn.get_accuracy(train_set, train_label)
        acc_before_test = self.qnn.get_accuracy(test_set, test_label)
        df = self.qnn.train(train_set, train_label, test_set, test_label, n_epochs={0: 10, 1: 30}, silent=False)
        print('Final stats')
        print(df)

        # Assert that bias has been modified
        self.assertFalse(np.array_equal(self.qnn.params, init_params))
        self.assertFalse(np.array_equal(self.qnn.projection_angles, init_projection_params))

        # Assert stats improved
        acc_after = self.qnn.get_accuracy(train_set, train_label)
        acc_after_test = self.qnn.get_accuracy(test_set, test_label)
        self.assertGreater(acc_after, acc_before)
        self.assertLess(df.iloc[-1]['loss'], df.iloc[0]['loss'])
        if acc_after_test < acc_before_test:
            warn('Accuracy of test set has decreased', UserWarning, stacklevel=2)

    @unittest.skip('Parallel execution does not work as expected')
    def test_train_parallel(self):
        # run on simpler dataset
        train_set, train_label = Sampler.circle(n_points=20)
        test_set, test_label = Sampler.circle(n_points=10)
        qnn_parallel = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, n_workers=4)
        df = qnn_parallel.train(train_set, train_label, test_set, test_label, silent=False)
        print('Final stats')
        print(df)

        # Run in parallel
        qnn_parallel = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, n_workers=10)
        start = time()
        df_parallel = qnn_parallel.train(train_set, train_label, test_set, test_label, silent=False).block_until_ready()
        t_parallel = time() - start

        # Run sequential
        start = time()
        df_sequential = self.qnn.train(train_set, train_label, test_set, test_label, silent=False).block_until_ready()

        t_sequential = time() - start

        # Test results and times
        ic(t_parallel, t_sequential)
        pd.testing.assert_frame_equal(df_parallel, df_sequential)
        self.assertLess(t_parallel, t_sequential)

    def test_train_without_test(self):
        df = self.qnn.train(self.train_set, self.train_label, silent=False)
        print('Final stats')
        print(df)

    def test_save_and_load(self):
        path = 'data/test_save_and_load_gate'

        qnn = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, n_workers=1)
        qnn.train(self.train_set, self.train_label, silent=False)
        qnn.save_qnn(path)

        qnn_loaded = GateQNN.load_qnn(path)
        self.assertEqual(qnn.training_info, qnn_loaded.training_info)

    def test_loaded_qnn_is_able_to_predict(self):
        path = 'data/test_save_and_load_gate'
        qnn = GateQNN.load_qnn(path)
        accuracy = qnn.get_accuracy(self.test_set, self.test_label)
        ic(accuracy)


    def test_train_interface_pennylane(self):
        train_set, train_label = Sampler.circle(n_points=20, interface='pennylane')
        qnn = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, n_workers=1, interface='pennylane')
        qnn.train(train_set, train_label, silent=False)

    def test_train_interface_pennylane_mnist(self):
        train_set, train_label, _, _ = MNISTSampler.fashion(20,  10, points_dimension=3, interface='pennylane')
        train_label = train_label.astype(int)
        qnn = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, n_workers=1, interface='pennylane')
        qnn.train(train_set, train_label, silent=False)

    def test_train_cost_pennylane_mnist(self):
        train_set, train_label, _, _ = MNISTSampler.fashion(20,  10, points_dimension=3, interface='pennylane')
        train_label = train_label.astype(int)
        qnn = GateQNN(num_qubits=1, num_layers=3, n_workers=1, interface='pennylane')
        cost = qnn.cost_qml(qnn.params, 0, train_set, train_label)
        ic(cost)


    def test_prediction_shape(self):
        data_points_train_ = increase_dimensions(self.train_set, 3)
        predicted_train = self.qnn.forward(data_points_train_, up_to_qubit=self.qnn.num_qubits-1)
        self.assertEqual(predicted_train.shape, self.train_label.shape)

    def test_noise(self):
        ideal_qnn = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, noise=False, seed=42)
        ideal_cost = ideal_qnn.cost([ideal_qnn.params, ideal_qnn.projection_angles], self.train_set, self.train_label)
        noisy_qnn = GateQNN(num_qubits=self.n_qubits, num_layers=self.n_layers, noise=True, seed=42)
        noisy_cost = noisy_qnn.cost([ideal_qnn.params, ideal_qnn.projection_angles], self.train_set, self.train_label)
        # Check that noisy version is not the same as the idealized case (using same parameters)
        ic(ideal_cost, noisy_cost)
        self.assertNotEqual(ideal_cost, noisy_cost)
        self.assertLessEqual(noisy_cost, 1)
        self.assertGreaterEqual(noisy_cost, -1)

    def test_noisy_training(self):
        ideal_qnn = GateQNN(num_qubits=1, num_layers=3, noise=False, seed=42)
        noisy_qnn = GateQNN(num_qubits=1, num_layers=3, noise=True, seed=42)
        # Assert start from the same initial parameters
        noisy_qnn.params = ideal_qnn.params
        # Train both models
        print('Training ideal model')
        df_ideal = ideal_qnn.train(self.train_set, self.train_label, silent=False, n_epochs=3)
        print('================================================================================================')
        print('Training noisy model')
        df_noisy = noisy_qnn.train(self.train_set, self.train_label, silent=False, n_epochs=3)
        # Check that the weights have changed
        ic(ideal_qnn.params, noisy_qnn.params)
        self.assertFalse(np.array_equal(ideal_qnn.params, noisy_qnn.params))

    def test_transpile_CROT(self):

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit_crot(angles, init):
            qml.StatePrep(init, wires=[0, 1])
            qml.CRot(phi=angles[0], theta=angles[1], omega=angles[2], wires=[0, 1])
            return qml.state()

        @qml.qnode(dev)
        def circuit_transpiled(angles, init):
            qml.StatePrep(init, wires=[0, 1])
            # A
            qml.RZ(angles[0], wires=1)
            qml.RY(angles[1]/2, wires=1)
            # CNOT
            qml.CNOT(wires=[0, 1])
            # B
            qml.RY(-angles[1]/2, wires=1)
            qml.RZ(-(angles[0] + angles[2])/2, wires=1)
            # CNOT
            qml.CNOT(wires=[0, 1])
            # C
            qml.RZ(-(angles[0] - angles[2])/2, wires=1)
            return qml.state()

        key = jax.random.PRNGKey(42)
        angles = jax.random.uniform(key, shape=(3,))
        init_state = jax.random.uniform(key, shape=(2**2,))
        init_state /= np.linalg.norm(init_state)

        state_crot = circuit_crot(angles, init_state)
        state_transpiled = circuit_transpiled(angles, init_state)

        ic(state_crot, state_transpiled)
        for s_crot, c_transpiled in zip(state_crot, state_transpiled):
            self.assertAlmostEqual(s_crot, c_transpiled, places=3)


if __name__ == '__main__':
    unittest.main()

from icecream import ic
import unittest
import numpy as np
from src.Sampler import RandomSampler  # Import your RandomSampler class here
from src.visualization import plot_2d_dataset
from src.utils import  accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Define simple classic models
def train_and_evaluate_classic_model(x_train, y_train, x_test=None, y_test=None):
    classic_model = RandomForestClassifier()
    classic_model.fit(x_train, y_train)

    pred_train = classic_model.predict(x_train)
    train_accuracy = accuracy_score(y_train, pred_train)

    if x_test is not None and y_test is not None:
        pred_test = classic_model.predict(x_test)
        test_accuracy = accuracy_score(y_test, pred_test)
        return train_accuracy, test_accuracy
    else:
        return train_accuracy


def model_pablo(x_train, y_train, x_test=None, y_test=None):
    clf = SVC(kernel='rbf')
    clf.fit(x_train, y_train)

    train_labels_pred = clf.predict(x_train)
    test_labels_pred = clf.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_labels_pred)
    if x_test is not None and y_test is not None:
        test_accuracy = accuracy_score(y_test, test_labels_pred)
        return train_accuracy, test_accuracy
    else:
        return train_accuracy

class TestRandomSampler(unittest.TestCase):

    def test_get_data_shapes(self):
        # Test if the function returns data with correct shapes
        x_train, y_train, x_test, y_test = RandomSampler.get_data(dimension=3, n_train=1000, n_test=1000, seed=42)

        # Check that the training and testing sets have the correct shape
        self.assertEqual(x_train.shape, (1000, 3))  # 1000 samples, reduced to 3 dimensions
        self.assertEqual(y_train.shape, (1000,))
        self.assertEqual(x_test.shape, (1000, 3))  # 1000 samples, reduced to 3 dimensions
        self.assertEqual(y_test.shape, (1000,))

    def test_get_data_class_distribution(self):
        # Test if the function generates a balanced class distribution
        x_train, y_train, x_test, y_test = RandomSampler.get_data(dimension=3, n_train=1000, n_test=1000, seed=42)

        # Check that the classes are roughly balanced
        unique_train_classes, counts_train = np.unique(y_train, return_counts=True)
        unique_test_classes, counts_test = np.unique(y_test, return_counts=True)

        self.assertEqual(len(unique_train_classes), 2)
        self.assertEqual(len(unique_test_classes), 2)
        self.assertTrue(np.all(counts_train > 400))  # Ensuring a balanced dataset
        self.assertTrue(np.all(counts_test > 400))  # Ensuring a balanced dataset

    def test_reduce_dimension(self):
        # Test if the reduce_dimension function works correctly
        x_train, y_train, x_test, y_test = RandomSampler.get_data(dimension=3, n_train=1000, n_test=1000, seed=42)

        # Check if the reduced data has the correct dimensionality
        self.assertEqual(x_train.shape[1], 3)
        self.assertEqual(x_test.shape[1], 3)

    def test_train_test_split(self):
        # Test if train_test_split works as expected
        x_train, y_train, x_test, y_test = RandomSampler.get_data(dimension=3, n_train=1000, n_test=1000, seed=42)

        # Check that the data is properly split into train and test sets
        self.assertEqual(x_train.shape[0], 1000)
        self.assertEqual(x_test.shape[0], 1000)
        self.assertEqual(y_train.shape[0], 1000)
        self.assertEqual(y_test.shape[0], 1000)

    def test_visualize_dataset(self):
        x_train, y_train, _, _ = RandomSampler.get_data(dimension=2,
                                                        n_informative=1,
                                                        n_redundant=0,
                                                        n_train=1000, n_test=1, seed=42)
        plot_2d_dataset(x_train, y_train, title="One informative feature")

        x_train, y_train, _, _ = RandomSampler.get_data(dimension=2,
                                                        n_informative=2,
                                                        n_redundant=0,
                                                        n_train=1000, n_test=1, seed=42)
        plot_2d_dataset(x_train, y_train, title="Two informative features")

        x_train, y_train, _, _ = RandomSampler.get_data(dimension=2,
                                                        n_features=30,
                                                        n_informative=10,
                                                        n_redundant=10,
                                                        n_cluster_per_class=5,
                                                        n_train=1000, n_test=1, seed=42)
        plot_2d_dataset(x_train, y_train, title="5 clusters per feature")

        x_train, y_train, _, _ = RandomSampler.get_data(dimension=2,
                                                        n_features=30,
                                                        n_informative=10,
                                                        n_redundant=10,
                                                        n_cluster_per_class=1,
                                                        n_train=1000, n_test=1, seed=42)
        plot_2d_dataset(x_train, y_train, title="1 clusters per feature")

    def test_visualize_easy_problem(self):
        x_train, y_train, x_test, y_test = RandomSampler.easy_problem(2, n_train=5000, n_test=100)
        plot_2d_dataset(x_train, y_train, title="Easy problem")
        # Solve with classifical models
        train_accuracy, test_accuracy = train_and_evaluate_classic_model(x_train, y_train, x_test, y_test)
        ic(train_accuracy, test_accuracy)
        self.assertGreaterEqual(train_accuracy, 0.97)
        self.assertGreaterEqual(test_accuracy, 0.95)
    def test_visualize_medium_problem(self):
        x_train, y_train, x_test, y_test = RandomSampler.medium_problem(2, n_train=5000, n_test=100)
        plot_2d_dataset(x_train, y_train, title="Medium problem")
        # Solve with classifical models
        train_accuracy, test_accuracy = train_and_evaluate_classic_model(x_train, y_train, x_test, y_test)
        ic(train_accuracy, test_accuracy)
        self.assertGreaterEqual(train_accuracy, 0.9)
        self.assertGreaterEqual(test_accuracy, 0.75)
    def test_visualize_hard_problem(self):
        x_train, y_train, x_test, y_test = RandomSampler.hard_problem(2, n_train=5000, n_test=100)
        plot_2d_dataset(x_train, y_train, title="Hard problem")
        # Solve with classifical models
        train_accuracy, test_accuracy = train_and_evaluate_classic_model(x_train, y_train, x_test, y_test)
        ic(train_accuracy, test_accuracy)
        self.assertGreaterEqual(train_accuracy, 0.8)
        self.assertGreaterEqual(test_accuracy, 0.6)


if __name__ == '__main__':
    unittest.main()

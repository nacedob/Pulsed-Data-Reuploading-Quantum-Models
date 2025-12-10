from src.experiments.config_exp import get_dataset
from icecream import ic
from warnings import filterwarnings
filterwarnings('ignore', category=FutureWarning)
filterwarnings('ignore', category=DeprecationWarning)


def check_dataset(dataset: str) -> bool:
    params = {
        'n_train': 20,
        'n_test': 10,
        'seed': 0,
        'points_dimension': 3,
        'interface': 'jax'
    }
    params.update(dataset=dataset)
    train_set_1, train_labels_1, test_set_1, test_labels_1 = get_dataset(**params)
    train_set_2, train_labels_2, test_set_2, test_labels_2 = get_dataset(**params)
    if not ((train_set_1 == train_set_2).all()
            and (train_labels_1 == train_labels_2).all()
            and (test_set_1 == test_set_2).all()
            and (test_labels_1 == test_labels_2).all()
            ):
        return False

    return True


def test_spiral():
    assert check_dataset('spiral')


def test_circles():
    assert check_dataset('circles')


def test_shell():
    assert check_dataset('shell')


def test_helix():
    assert check_dataset('helix')


def test_iris():
    assert check_dataset('iris')


def test_digits():
    assert check_dataset('digits')


def test_sinus():
    assert check_dataset('sinus')


def test_sinus3d():
    assert check_dataset('sinus3d')


def test_fashion():
    assert check_dataset('fashion')

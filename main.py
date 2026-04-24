"""
This is a simple example.
You can find more examples in the examples folder
"""
from typing import Literal
from src.QNN import PulsedQNN, GateQNN
from src.experiments.config_exp import get_dataset
from warnings import filterwarnings

filterwarnings('ignore', category=FutureWarning)
filterwarnings('ignore', category=RuntimeWarning)
filterwarnings('ignore', category=DeprecationWarning)

N_QUBITS = 1
N_LAYERS = 3
N_TRAIN = 300
N_TEST = 100
SEED = 42

def train(
    model: Literal['pulsed', 'gate'],
    dataset: str
) -> None:
    # 1. Define the QNN
    qnn = (
        PulsedQNN(num_qubits=N_QUBITS, num_layers=N_LAYERS, seed=SEED, encoding='gate') if model == 'gate' else 
        GateQNN(num_qubits=N_QUBITS, num_layers=N_LAYERS, seed=SEED)
    )
    # 2. Load dataset
    X_train, y_train, X_test, y_test = get_dataset(
        dataset=dataset, n_train=N_TRAIN, n_test=N_TEST, interface='jax', points_dimension=3, seed=SEED
    )
    # 3. Train the QNN
    print(f'Training the {model.upper()} QNN...')
    qnn.train(X_train, y_train, X_test, y_test, silent=True)
    # 4. Print final accuracies
    print('Final accuracies:')
    print('Train loss: ', qnn.cost(X_train, y_train))
    print('Test loss: ', qnn.cost(X_test, y_test))
    print(f"Train accuracy: {qnn.get_accuracy(X_train, y_train):.4f}")
    print(f"Test accuracy: {qnn.get_accuracy(X_test, y_test):.4f}")
    

if __name__ == '__main__':
    # Run some examples
    for dataset in ['iris', 'digits_17', 'digits_56', 'corners3d']:  
        print(f"\n{'='*30}\nDATASET: {dataset}\n{'='*30}")
        for model in ['pulsed', 'gate']:
            train(model=model, dataset=dataset)
        print(f"\n{'='*30}\n")
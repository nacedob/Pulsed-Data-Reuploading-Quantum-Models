import sys
from ast import literal_eval
from typing import List, Iterable
from warnings import filterwarnings
from icecream import ic

from src.experiments.final_experiment import main
from config import DATASETS

filterwarnings('ignore', category=RuntimeWarning)
filterwarnings('ignore', category=Warning)

# --- Global Constants ---
SEEDS = 1
JOBS = 3
DEBUG_NOISE = False
METRIC_TUNING = 'loss'
N_TRAIN = 300
N_TEST = 100
EXPERIMENT_FOLDER = 'LAYERS_EXPERIMENT'
NOISE_SOURCES = ['all']

# Default Experiment Dictionary (Template)
BASE_ARGS = {
    'models': "['gate', 'mixed']",
    'n_qubits': 2,
    'n_seeds': 1,
    'layers_min': 0,
    'layers_max': 50,
    'layers_step': 5,
    'n_epochs': 30,
    'trials_tuning': 30,
    'n_jobs': JOBS,
    'tuning': True,
    'noise': True,
    'load': True,
    'optimizer': 'rms',
    'save_qnn': False,
    'metric_tuning': METRIC_TUNING,
    'n_train': N_TRAIN,
    'n_test': N_TEST,
    'folder': EXPERIMENT_FOLDER,
    'regularization': 0,
    'debug_noise': DEBUG_NOISE
}


def parse_runtime_args() -> tuple[List[str], Iterable[int]]:
    models = ['gate', 'mixed']
    seeds = range(SEEDS)

    if len(sys.argv) > 1:
        try:
            models = literal_eval(sys.argv[1])
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse model list '{sys.argv[1]}'. Using defaults.")

    if len(sys.argv) > 2:
        try:
            seeds = literal_eval(sys.argv[2])
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse seed list '{sys.argv[2]}'. Using defaults.")

    return models, seeds


if __name__ == '__main__':
    model_list, seed_list = parse_runtime_args()

    for dataset in DATASETS:
        for seed in seed_list:
            # Prepare overrides for this specific iteration
            iteration_overrides = BASE_ARGS.copy()
            iteration_overrides.update({
                'dataset': dataset,
                'starting_seed': seed,
                'trained_models': model_list
            })

            print("\n" + "=" * 50)
            ic(dataset, seed, model_list)
            print("=" * 50 + "\n")

            main(iteration_overrides)

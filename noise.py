

import numpy as np
from icecream import ic
from warnings import filterwarnings

# Internal Imports
from src.utils import load_json_to_dict
from src.experiments.final_experiment import main

# Global Settings
filterwarnings('ignore', category=RuntimeWarning)
filterwarnings('ignore', category=Warning)

# --- Configuration Constants ---
JOBS = 4
MODELS = ['gate', 'mixed']
METRIC_TUNING = 'loss'
N_TRAIN = 300
N_TEST = 100
DEBUG_NOISE = False
LOAD = True
REGULARIZATION_QNN = 0
EXPERIMENT_FOLDER = 'NOISE_EXPERIMENT'

# Noise Setup
noise_mapper = load_json_to_dict('data/backends/custom_mapper.json')
LOW_NOISE = str(noise_mapper['low_noise']).replace(' ', '')
MIDDLE_NOISE = str(noise_mapper['middle_noise']).replace(' ', '')
HIGH_NOISE = str(noise_mapper['high_noise']).replace(' ', '')

# Generate noise range: [0, 0.03, 0.06, ..., 0.3]
NOISE_VALUES = np.arange(0, 0.3001, 0.03).round(3)
DATASETS = ['digits_08']

def run_orchestrator():
    # We iterate seeds first as per your original logic requirements
    for seed in range(6):
        print(f"\n{'='*30}\nRUNNING BATCH FOR SEED: {seed}\n{'='*30}")
        
        for dataset in DATASETS:
            for p in NOISE_VALUES:
                p_val = float(p)
                
                # Construct the overrides dictionary
                # Keys match the DEFAULTS in final_experiment.py
                experiment_config = {
                    'models': "['gate', 'mixed']",
                    'dataset': dataset,
                    'n_qubits': 2,
                    'layers_min': 20,
                    'layers_max': 21,
                    'layers_step': 5,
                    'noise': True,
                    'n_seeds': 1,
                    'starting_seed': seed,
                    'n_jobs': JOBS,
                    'trained_models': MODELS,
                    # Pass the dict directly, main() handles the rest
                    'noise_parameters': {'depolarizing_1q': p_val},
                    'metric_tuning': METRIC_TUNING,
                    'n_train': N_TRAIN,
                    'n_test': N_TEST,
                    'regularization': REGULARIZATION_QNN,
                    'debug_noise': DEBUG_NOISE,
                    'load': LOAD,
                    'tuning': True,
                    'optimizer': 'rms',
                    'folder': EXPERIMENT_FOLDER,
                    'n_epochs': 30,
                    'trials_tuning': 30,
                }

                ic(dataset, seed, p_val)
                
                try:
                    main(overrides=experiment_config)
                    ic(f"[INFO] Finished Dataset: {dataset} | Seed: {seed} | Noise: {p_val}")
                except Exception as e:
                    ic(f"[ERROR] Experiment failed: {e}")
                    # Continue to next experiment even if one fails
                    continue
                finally:
                    print(f"\n{'='*30}\n")

if __name__ == '__main__':
    run_orchestrator()
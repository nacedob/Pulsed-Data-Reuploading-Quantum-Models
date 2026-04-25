"""
This is the final experiment.
There are lots of options so one can perform almost any experiement he wants
The idea is to get the dataset, tune the model and choose the best learning rate. Then get stats for the best
model posible, and iterate over the available models (gate, gate_spherical, mixed, pulsed).
One can iterate as well over the number of qubits or the number of layers
The noise parameters can be also changed
"""

from icecream import ic
import os
import ast
import argparse
import numpy as np
import pandas as pd
from time import time
from typing import Optional, Dict, Any, Tuple
from collections import defaultdict
from warnings import filterwarnings

# Internal Imports
from src.utils import (
    save_dict_to_json,
    print_in_blue,
    pickle_extension
)
from src.QNN.BaseQNN import BaseQNN
from config import get_root_path
from .config_exp import (
    get_highest_id,
    get_dataset,
    get_qnn,
    get_stats,
    _get_device_folder
)

import optuna

# Global Silencing
filterwarnings('ignore', category=FutureWarning)
filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration Constants ---
LR_MIN, LR_MAX = 0.00005, 0.05
BATCH_SIZE = 24
DEFAULT_LRS = {'gate': 0.05, 'mixed': 0.00045}

# --- Single Source of Truth for Defaults ---
DEFAULTS = {
    'layers_min': 1,
    'layers_max': 10,
    'layers_step': 1,
    'n_qubits': 'all',
    'tuning': True,
    'trials_tuning': 50,
    'metric_tuning': 'loss',
    'n_jobs': 4,
    'n_seeds': 5,
    'starting_seed': 0,
    'n_train': 500,
    'n_test': 250,
    'n_epochs': 30,
    'point_dimension': 3,
    'load': False,
    'noise': False,
    'dataset': 'fashion',
    'trained_models': 'all',
    'debug_noise': False,
    'optimizer': 'rms',
    'interface': 'jax',
    'save_qnn': False,
    'save_data': False,
    'lr': 0.0,
    'regularization': 0.0,
    'realistic_gates': False,
    'eqk': False,
    'use_stored_tuning': False,
    'noise_parameters': None,
    'noise_sources': 'all',
    'folder': None
}


def str_to_bool(value: Any) -> bool:
    """Safely convert strings or booleans to boolean."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', 'yes', 't', '1', 'y')


def get_parser() -> argparse.ArgumentParser:
    """Generates the argument parser based on the DEFAULTS dictionary."""
    parser = argparse.ArgumentParser(description="Quantum Data Reuploading Experiment")
    for key, val in DEFAULTS.items():
        arg_name = f"--{key}"
        if isinstance(val, bool):
            parser.add_argument(arg_name, type=str_to_bool, default=val)
        elif isinstance(val, int):
            parser.add_argument(arg_name, type=int, default=val)
        elif isinstance(val, float):
            parser.add_argument(arg_name, type=float, default=val)
        else:
            parser.add_argument(arg_name, type=str, default=val)
    return parser

def get_partial_path(folder: str, n_qubits: int, n_layers: int, seed: int) -> str:
    return f'{folder}/partial/n_qubits_{n_qubits}__n_layers_{n_layers}__seed_{seed}.csv'


def get_tuning_score(
    train_set: np.ndarray,
    train_labels: np.ndarray,
    test_set: np.ndarray,
    test_labels: np.ndarray,
    model: str,
    seed: int,
    qubits: int,
    n_layers: int,
    n_epochs: int,
    batch_size: int,
    opt_params: Dict[str, float],
    config: Dict[str, Any],
    tuning_dfs: Dict[str, pd.DataFrame],
    tuned_qnn: Dict[str, Tuple[Any, float, float]]
) -> float:

    qnn = get_qnn(
        model=model,
        n_qubits=qubits,
        n_layers=n_layers,
        realistic_gates=config['realistic_gates'],
        seed=seed,
        interface=config['interface'],
        noise=config['noise'],
        debug_noise=config['debug_noise'],
        noise_parameters=config['noise_parameters'],
        regularization=config['regularization'],
        noise_sources=config['noise_sources_list']
    )

    qnn.train(
        data_points_train=train_set,
        data_labels_train=train_labels,
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer=config['optimizer'],
        optimizer_parameters=opt_params,
        silent=True,
        save_stats=False
    )

    train_loss, final_acc_train, test_loss, final_acc_test = get_stats(
        qnn=qnn,
        train_set=train_set,
        train_labels=train_labels,
        test_set=test_set,
        test_labels=test_labels
    )

    if config['tuning']:
        stat_row = pd.DataFrame([{
            'n_qubits': qubits, 'n_layers': n_layers, 'seed': seed,
            'metric': config['metric_tuning'], 'train_loss': float(train_loss),
            'test_loss': float(test_loss), 'acc_train': float(final_acc_train),
            'acc_test': float(final_acc_test), 'lr': float(opt_params['lr']),
        }])
        tuning_dfs[model] = pd.concat([tuning_dfs[model], stat_row], ignore_index=True)

        if train_loss < tuned_qnn[model][1]:
            test_cost = qnn.cost([qnn.params, qnn.projection_angles], test_set, test_labels)
            tuned_qnn[model] = (qnn, train_loss, test_cost)

    return float(train_loss) if config['metric_tuning'] == 'loss' else -(2 * final_acc_train + final_acc_test) / 3


def train_and_evaluate(
    qnn: BaseQNN,
    exp_id: int,
    train_set: np.ndarray,
    train_labels: np.ndarray,
    test_set: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
    n_qubits: int,
    n_layers: int,
    config: Dict[str, Any],
    QNN_PATH: str
) -> pd.DataFrame:

    print(f'[TRAINING] Training model {qnn.model_name}')
    qnn.train(
        data_points_train=train_set,
        data_labels_train=train_labels,
        n_epochs=config['n_epochs'],
        batch_size=BATCH_SIZE,
        optimizer_parameters={'lr': config['current_lr']},
        silent=True,
        save_stats=False
    )

    train_loss, final_acc_train, test_loss, final_acc_test = get_stats(
        qnn=qnn,
        train_set=train_set,
        train_labels=train_labels,
        test_set=test_set,
        test_labels=test_labels
    )

    model_key = 'gate'
    if 'spherical' in qnn.model_name:
        model_key = 'gate_spherical'
    elif 'encoding_gate' in qnn.model_name:
        model_key = 'mixed'
    elif 'encoding_pulsed' in qnn.model_name:
        model_key = 'pulsed'

    if config['save_qnn']:
        qnn_folder = f"{QNN_PATH}/{model_key}/exp_{exp_id}"
        os.makedirs(qnn_folder, exist_ok=True)
        save_path = os.path.join(qnn_folder, f'qnn_q_{n_qubits}_l_{n_layers}_s_{seed}.{pickle_extension}')
        qnn.save_qnn(path=save_path)
        
    print(f'\n Model {model_key} trained for n_qubits={n_qubits}, n_layers={n_layers}, seed={seed}, dataset={config["dataset"]} \n ')
    print(f"\tTrain loss: {train_loss}")
    print(f"\tTest loss: {test_loss}")
    print(f"\tTrain accuracy: {final_acc_train}")
    print(f"\tTest accuracy: {final_acc_test}\n")

    return pd.DataFrame([{
        'model': model_key, 'n_qubits': n_qubits, 'n_layers': n_layers, 'seed': seed,
        'train_loss': float(train_loss), 'test_loss': float(test_loss),
        'acc_train': float(final_acc_train), 'acc_test': float(final_acc_test),
        'lr': config['current_lr'], 'dataset': config['dataset']
    }]).set_index('n_layers')


def main(overrides: Optional[Dict[str, Any]] = None):
    # 1. Resolve Config
    config = DEFAULTS.copy()
    if overrides:
        config.update(overrides)
    else:
        args = get_parser().parse_args()
        config.update(vars(args))

    # 2. Logic Cleanup
    if isinstance(config['noise_parameters'], str):
        config['noise_parameters'] = ast.literal_eval(config['noise_parameters'])

    src_str = str(config['noise_sources'])
    config['noise_sources_list'] = [
        s for s in [
            'depolarizing',
            'amplitude',
            'phase'] if s in src_str or src_str == 'all']

    if config['n_qubits'] == 'all':
        config['n_qubits'] = [1, 2]
    elif not isinstance(config['n_qubits'], list):
        config['n_qubits'] = [int(config['n_qubits'])]

    # 3. Pathing
    suffix = ('_noise' if config['noise'] else '') + ('_debug' if config['debug_noise'] else '')
    device_folder = _get_device_folder(
        noise_parameters=config['noise_parameters'],
        noise_sources=config['noise_sources_list']
    )

    root = get_root_path()
    results_path = os.path.join(
        root,
        f"data/results/{config['folder'] or 'final_experiment'+suffix}/{device_folder}/{config['dataset']}"
    )
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(f'{results_path}/partial', exist_ok=True)

    exp_id = get_highest_id(folder_path=results_path) + 1
    print_in_blue(text=f"Starting Experiment ID: {exp_id} | Dataset: {config['dataset']}")

    all_stats = []
    start_time = time()

    models = ['gate', 'gate_spherical', 'mixed', 'pulsed']
    if config['trained_models'] != 'all':
        models = [m for m in models if m in config['trained_models']]
    
    # 4. Loops
    for n_q in config['n_qubits']:
        for n_l in range(config['layers_min'], config['layers_max'] + 1, config['layers_step']):
            if n_l == 0:
                n_l = 1
            for seed in range(config['starting_seed'], config['starting_seed'] + config['n_seeds']):
                print('='*80)
                print(f'Dataset: {config["dataset"]} | Qubits: {n_q} | Layers: {n_l} | Seed: {seed}')
                
                partial_path = get_partial_path(folder=results_path, n_qubits=n_q, n_layers=n_l, seed=seed)
                
                # Skip if already exists
                if os.path.exists(partial_path):
                    stats_df = pd.read_csv(partial_path)
                    models_in_partial = set(stats_df['model'].unique())
                    
                    if set(models).issubset(models_in_partial):
                        print(f'Loaded from {partial_path}\n\n')
                        partial_results_df = stats_df[stats_df['model'].isin(models)]
                        continue
                print('Starting experiment')

                train_x, train_y, test_x, test_y = get_dataset(
                    dataset=config['dataset'],
                    n_train=config['n_train'],
                    n_test=config['n_test'],
                    interface=config['interface'],
                    points_dimension=config['point_dimension'],
                    seed=seed
                )

                partial_results_df = pd.DataFrame()
                
                for model_type in models:
                    if config['tuning']:
                        study = optuna.create_study(direction="minimize")
                        study.optimize(
                            lambda t: get_tuning_score(
                                train_set=train_x,
                                train_labels=train_y,
                                test_set=test_x,
                                test_labels=test_y,
                                model=model_type,
                                seed=seed,
                                qubits=n_q,
                                n_layers=n_l,
                                n_epochs=config['n_epochs'],
                                batch_size=BATCH_SIZE,
                                opt_params={'lr': t.suggest_float('lr', LR_MIN, LR_MAX, log=True)},
                                config=config,
                                tuning_dfs=defaultdict(pd.DataFrame),
                                tuned_qnn={model_type: (None, float('inf'), 0)}
                            ), 
                            n_trials=config['trials_tuning'], 
                            n_jobs=config['n_jobs']
                        )
                        config['current_lr'] = study.best_params['lr']
                    else:
                        config['current_lr'] = config['lr'] or DEFAULT_LRS.get(model_type, 0.01)

                    qnn_instance = get_qnn(
                        model=model_type, n_qubits=n_q, n_layers=n_l,
                        realistic_gates=config['realistic_gates'], seed=seed,
                        interface=config['interface'], noise=config['noise'],
                        noise_parameters=config['noise_parameters']
                    )

                    stats_df = train_and_evaluate(
                        qnn=qnn_instance, exp_id=exp_id, train_set=train_x, train_labels=train_y,
                        test_set=test_x, test_labels=test_y, seed=seed, n_qubits=n_q, n_layers=n_l,
                        config=config, QNN_PATH=os.path.join(results_path, 'trained_qnn')
                    )
                    all_stats.append(stats_df)
                    partial_results_df = pd.concat([partial_results_df, stats_df])
                    
                # Save partial results
                partial_results_df.to_csv(partial_path)
                print(
                    f'Experiment for dataset {config["dataset"]}, {n_q} qubits, {n_l} layers and seed {seed} '
                    f'saved to {partial_path}\n\n'
                )
                

    # 5. Save Results
    if all_stats:
        pd.concat(all_stats).to_csv(os.path.join(results_path, f"results_{exp_id}.csv"))

    print_in_blue(text=f"[END] Experiment {exp_id} finished in {time() - start_time:.2f}s")


if __name__ == '__main__':
    main()

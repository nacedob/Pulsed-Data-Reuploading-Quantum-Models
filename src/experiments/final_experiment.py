"""
This is the final experiment.
There are lots of options so one can perform almost any experiement he wants
The idea is to get the dataset, tune the model and choose the best learning rate. Then get stats for the best
model posible, and iterate over the available models (gate, gate_spherical, mixed, pulsed).
One can iterate as well over the number of qubits or the number of layers
The noise parameters can be also changed
"""
from src.utils import save_dict_to_json, print_in_blue, get_root_path, get_current_folder, load_json_to_dict
import json
import optuna
import pandas as pd
from icecream import ic
import re
from time import time
import argparse
import ast
import numpy as np
from src.utils import pickle_extension
from src.QNN.BaseQNN import BaseQNN
from .config_exp import get_highest_id, get_dataset, get_qnn, get_optimal_lr, get_optimal_opt_parameters, get_stats
import os
from warnings import filterwarnings, warn
from .config_exp import _get_device_folder
from typing import Optional, Union
filterwarnings('ignore', category=FutureWarning)
filterwarnings('ignore', category=RuntimeWarning)


################ TUNING PARAMS #################################
parser = argparse.ArgumentParser()
parser.add_argument('--layers_min', type=int, help='#layers min to be compared', required=False, default=1)
parser.add_argument('--layers_max', type=int, help='#layers max to be compared', required=False, default=10)
parser.add_argument('--layers_step', type=int, help='step layers', required=False, default=1)
parser.add_argument('--n_qubits', type=str, help='#qubits to use for both models', required=False, default='all')
parser.add_argument('--tuning', help='whether to tune model before each experiment', required=False, default='True')
parser.add_argument('--trials_tuning', type=float, required=False, default=50,
                    help='Number of trials to tune hyperparameters for each model')
parser.add_argument('--metric_tuning', type=str, required=False, default='loss')
parser.add_argument('--n_jobs', type=int, help='Number of simultaneous tunning', required=False, default=4)
parser.add_argument('--n_seeds', type=int, help='Number of seeds to perform experiment', required=False, default=5)
parser.add_argument('--starting_seed', type=int, help='starting seed', required=False, default=0)
parser.add_argument('--n_train', type=int, help='# points for train set', required=False, default=500)
parser.add_argument('--n_test', type=int, help='# points for test set', required=False, default=250)
parser.add_argument('--n_epochs', type=int, help='# epochs', required=False, default=30)
parser.add_argument('--point_dimension', type=int, help='Dimension of dataset', required=False, default=3)
parser.add_argument('--load', type=str, help='whether to load results', required=False, default='False')
parser.add_argument('--noise', type=str, required=False, default='False')
parser.add_argument('--dataset', type=str, help='dataset to be used', required=False, default='fashion')
parser.add_argument('--trained_models', type=str, help='model to be trained', required=False, default='all')
parser.add_argument('--debug_noise', type=str, help='debug noise with Adrians idea', required=False,
                    default='False')
parser.add_argument('--optimizer', type=str, help='optimizer to be used', required=False, default='rms')
parser.add_argument('--interface', type=str, help='interface', required=False, default='jax')
parser.add_argument('--save_qnn', type=str, help='whether save trained qnns',
                    required=False, default='False')
parser.add_argument('--save_data', type=str, help='whether save the exact data that was used for training',
                    required=False, default='False')
parser.add_argument('--lr', type=float, help='learning_rate', required=False, default=0)
parser.add_argument('--regularization', type=float, help='regualization for qnn cost function',
                    required=False, default=0)
parser.add_argument('--realistic_gates', type=str, help='whether to use realistic gates for GateQNN',
                    required=False, default='False')
parser.add_argument('--eqk', type=str, help='step further and train eqk',
                    required=False, default='False')
parser.add_argument('--use_stored_tuning', type=str, help='whether to stored tuning results for lr',
                    required=False, default='False')
parser.add_argument('--noise_parameters', type=str, help='The noise custom parameters to be used',
                    required=False, default=None)
parser.add_argument('--noise_sources', type=str, help='The noise sources to be used',
                    required=False, default='all')
parser.add_argument('--folder', type=str, help='folder to save results', required=False, default=None)
args, _ = parser.parse_known_args()

DEFAULT_ARGS = {
    'realistic_gates': ast.literal_eval(args.realistic_gates),
    'n_qubits': args.n_qubits,
    'layers_min': args.layers_min,
    'layers_max': args.layers_max,
    'layers_step': args.layers_step,
    'regularization': args.regularization,
    'metric_tuning': args.metric_tuning,
    'trained_models': args.trained_models,
    'n_trials': args.trials_tuning,
    'n_jobs': int(args.n_jobs),
    'n_train': int(args.n_train),
    'n_test': int(args.n_test),
    'n_epochs': int(args.n_epochs),
    'point_dimension': int(args.point_dimension),
    'n_seeds': int(args.n_seeds),
    'starting_seed': int(args.starting_seed),
    'dataset': args.dataset,
    'optimizer': args.optimizer,
    'interface': args.interface,
    'LOAD_RESULTS': ast.literal_eval(args.load.capitalize()),
    'save_qnn': ast.literal_eval(args.save_qnn.capitalize()),
    'save_data': ast.literal_eval(args.save_data.capitalize()),
    'debug_noise': ast.literal_eval(args.debug_noise.capitalize()),
    'tuning_bool': ast.literal_eval(args.tuning.capitalize()),
    'eqk_bool': ast.literal_eval(args.eqk.capitalize()),
    'noise_bool': ast.literal_eval(args.noise.capitalize()),
    'use_stored_tuning': ast.literal_eval(args.use_stored_tuning.capitalize()),
    'lr': args.lr,
    'noise_parameters_str': args.noise_parameters or str({}),
    'noise_sources_str': args.noise_sources,
    'experiment_folder': args.folder,
}


################ TUNING GRID #################################
DEFAULT_LRS = {'gate': 0.05, 'mixed': 0.00045}

LR_MIN, LR_MAX = 0.00005, 0.05
BETA1 = 0.9
BETA2 = 0.999
BATCH_SIZE = 24


################ TUNING EXPERIMENT FUNCTIONS #################################
def get_tuning_score(train_set, train_labels, test_set, test_labels,
                     model: str, seed: int, qubits: int, n_layers: int, n_epochs: int, batch_size: int,
                     opt_params: dict, dataset: str, realistic_gates: bool, metric_tuning: str,
                     tuning_bool: bool,
                     noise_bool: bool, debug_noise: bool, noise_parameters: Optional[dict],
                     regularization: float, noise_sources: list[str],
                     optimizer: str, interface: str, tuning_dfs: dict[str, pd.DataFrame],
                     tuned_qnn: dict[str, tuple[Optional[BaseQNN], float, float]],):
    print(f'Starting trial with: {dataset=}, {qubits=}, {n_layers=}, {seed=}, {n_epochs=}, {opt_params=}')

    qnn = get_qnn(model=model, n_qubits=qubits, n_layers=n_layers, realistic_gates=realistic_gates, seed=seed,
                  interface=interface, noise=noise_bool, debug_noise=debug_noise, noise_parameters=noise_parameters,
                  regularization=regularization, noise_sources=noise_sources)

    df_train = qnn.train(data_points_train=train_set, data_labels_train=train_labels,
                         # data_points_test=test_set, data_labels_test=test_labels,     # Para que vaya mas rapido
                         n_epochs=n_epochs,
                         batch_size=batch_size,
                         optimizer=optimizer,
                         optimizer_parameters=opt_params,
                         silent=True,  # To go faster
                         save_stats=False,  # To go faster
                         )

    train_loss, final_acc_train, test_loss, final_acc_test = get_stats(
        qnn, train_set, train_labels, test_set, test_labels)

    # Save qnn
    stat_row = pd.DataFrame([{
        'n_qubits': qubits,
        'n_layers': n_layers,
        'seed': seed,
        'metric': metric_tuning,
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'acc_train': float(final_acc_train),
        'acc_test': float(final_acc_test),
        'lr': float(opt_params['lr']),
    }])
    if tuning_bool:
        tuning_dfs[model] = pd.concat([tuning_dfs[model], stat_row], ignore_index=True)
        current_cost = tuned_qnn[model][1]
        if train_loss < current_cost:
            test_cost = qnn.cost([qnn.params, qnn.projection_angles], test_set, test_labels)
            tuned_qnn[model] = (qnn, current_cost, test_cost)

    if metric_tuning == 'loss':
        return float(train_loss)
    elif metric_tuning == 'combination_acc':
        return -(2 * final_acc_train + final_acc_test) / 3
    else:
        raise ValueError(f'{metric_tuning} not recognized')


def objective_tuning(trial, train_set, train_labels, test_set, test_labels,
                     seed: int, qubits: int, n_layers: int, model: str, n_epochs: int,
                     dataset, realistic_gates, metric_tuning,
                     tuning_bool, noise_bool, debug_noise, noise_parameters,
                     regularization, noise_sources,
                     optimizer, interface,
                     tuning_dfs: dict[str, pd.DataFrame], tuned_qnn: dict[str, tuple[Optional[BaseQNN], float, float]]):
    lr_ = trial.suggest_float('lr', LR_MIN, LR_MAX, log=True)
    opt_params = {'lr': lr_}
    score = get_tuning_score(
        train_set=train_set,
        train_labels=train_labels,
        test_set=test_set,
        test_labels=test_labels,
        batch_size=BATCH_SIZE,
        model=model,
        seed=seed,
        qubits=qubits,
        n_layers=n_layers,
        n_epochs=n_epochs,
        opt_params=opt_params,
        dataset=dataset,
        realistic_gates=realistic_gates,
        metric_tuning=metric_tuning,
        tuning_bool=tuning_bool,
        noise_bool=noise_bool,
        debug_noise=debug_noise,
        noise_parameters=noise_parameters,
        regularization=regularization,
        noise_sources=noise_sources,
        optimizer=optimizer,
        interface=interface,
        tuning_dfs=tuning_dfs,
        tuned_qnn=tuned_qnn,
    )
    return score


def tune_model(model: str, seed: int, qubits: int, n_layers: int, exp_id: int,
               n_trials: int, n_jobs: int, n_train: int, n_test: int,
               dataset: str, metric_tuning: str, point_dimension: int,
               n_epochs: int, realistic_gates: bool, optimizer: str, EXP_RESULTS_PATH: str,
               tuning_bool, noise_bool, debug_noise, noise_parameters,
                regularization, noise_sources, interface, train_set, train_labels, test_set, test_labels,
               tuning_dfs: dict[str, pd.DataFrame], tuned_qnn: dict[str, tuple[Optional[BaseQNN], float, float]]):
    print(f'[TUNING] Starting tuning experiment for model: {model.upper()}')
    study = optuna.create_study(direction="minimize", study_name=model)
    objective_gate = lambda trial: objective_tuning(
        trial=trial,
        train_labels=train_labels,
        train_set=train_set,
        test_labels=test_labels,
        test_set=test_set,
        n_epochs=n_epochs,
        seed=seed,
        qubits=qubits,
        n_layers=n_layers,
        model=model,
        dataset=dataset,
        realistic_gates=realistic_gates,
        metric_tuning=metric_tuning,
        tuning_bool=tuning_bool,
        noise_bool=noise_bool,
        debug_noise=debug_noise,
        noise_parameters=noise_parameters,
        regularization=regularization,
        noise_sources=noise_sources,
        optimizer=optimizer,
        interface=interface,
        tuning_dfs=tuning_dfs,
        tuned_qnn=tuned_qnn
    )
    study.optimize(objective_gate, n_trials=n_trials, n_jobs=n_jobs)
    exp_dict = {
        "model": model,
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "n_train": n_train,
        "n_test": n_test,
        "dataset": dataset,
        'metric': metric_tuning,
        "point_dimension": point_dimension,
        "n_layers": n_layers,
        "LR_MIN": LR_MIN,
        "LR_MAX": LR_MAX,
        "n_qubits": qubits,
        "BETA1": BETA1,
        "BETA2": BETA2,
        "n_epochs": n_epochs,
        "BATCH_SIZE": BATCH_SIZE,
        "realistic_gates": realistic_gates,
    }

    save_results_tuning(exp_id, qubits, n_layers, seed, study, exp_dict, model, optimizer, EXP_RESULTS_PATH)
    params_tuned = study.best_params
    print(f'[TUNING] Finished tuning experiment for model: {model.upper()}')
    return params_tuned


def get_model_from_qnn_name(qnn_name: str) -> str:
    MODEL_NAME_MAP = {
        'GateQNN': 'gate',
        'PulsedQNN_encoding_gate': 'mixed',
    }
    try:
        return MODEL_NAME_MAP[qnn_name]
    except KeyError:
        raise ValueError(f'Unrecognized QNN name: {qnn_name}')


################ TRAINING FUNCTIONS #################################
def train_and_evaluate(qnn, exp_id: Union[str, int], train_set, train_labels, test_set, test_labels, seed,
                       n_qubits: int, n_layers: int, batch_size: int, lr: float,
                       n_epochs: int, save_qnn, optimizer: str, dataset, realistic_gates: bool,
                       save_data, QNN_PATH: str):
    """
    This method creates a df containing information about the performance of the qnn after trainign it with
    the required characteristics. This df contains as keys:
       - model
       - n_qubits
       - n_layers
       - seed
       - train_loss
       - test_loss
       - acc_train
       - acc_test
       - lr
       - epochs
       - dataset
       - optimizer
       - qnn_path
    """
    print(f'[TRAINING] Training model {qnn.model_name}')

    df_train = qnn.train(data_points_train=train_set, data_labels_train=train_labels,
                         # data_points_test=test_set, data_labels_test=test_labels,     # Para que vaya mas rapido
                         n_epochs=n_epochs,
                         batch_size=batch_size,
                         optimizer_parameters={'lr': lr},
                         silent=True,  # To go faster
                         save_stats=False,  # To go faster
                         )

    train_loss, final_acc_train, test_loss, final_acc_test = get_stats(
        qnn, train_set, train_labels, test_set, test_labels)
    ic(final_acc_test, final_acc_train, train_loss, test_loss)

    if save_qnn:
        if qnn.model_name == 'GateQNN_spherical':
            model = 'gate_spherical'
        elif qnn.model_name == 'GateQNN':
            model = 'gate'
        elif qnn.model_name == 'PulsedQNN_encoding_gate':
            model = 'mixed'
        else:
            raise ValueError(f'{qnn.model_name} not recognized')
        qnn_path = get_qnn_path(model, n_qubits, n_layers, seed, exp_id, QNN_PATH)
        qnn.save_qnn(qnn_path)

        # save dict containing qnn parameters
        params_dict = {
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': optimizer,
            'final_accuracy_train': float(final_acc_train),
            'final_accuracy_test': float(final_acc_test),
            'train_loss': float(train_loss),
            'test_loss': float(test_loss),
            'seed': seed,
            'model': qnn.model_name,
            'dataset': dataset,
            "realistic_gates": realistic_gates,
        }
        save_dict_to_json(params_dict, qnn_path.replace(f'{pickle_extension}', 'json'))

        if save_data:
            datapath = get_data_path(dataset, model, exp_id, QNN_PATH)
            dataset_save = {
                'train_set': train_set.tolist(),
                'train_labels': train_labels.tolist(),
                'test_set': test_set.tolist(),
                'test_labels': test_labels.tolist(),
            }
            with open(datapath, "w") as f:
                json.dump(dataset_save, f, indent=4)

    else:
        qnn_path = None

    # Save stats to global df

    if qnn.model_name == 'GateQNN':
        model = 'gate'
    elif qnn.model_name == 'GateQNN_spherical':
        model = 'gate_spherical'
    elif qnn.model_name == 'PulsedQNN_encoding_gate':
        model = 'mixed'
    elif qnn.model_name == 'PulsedQNN_encoding_pulsed':
        model = 'pulsed'
    else:
        raise ValueError(f'{qnn.model_name} not recognized')

    qnn_stats = pd.DataFrame([{
        'model': model,
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'seed': seed,
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'acc_train': float(final_acc_train),
        'acc_test': float(final_acc_test),
        'lr': float(lr),
        'epochs': str(n_epochs),
        'dataset': dataset,
        'optimizer': optimizer,
        'qnn_path': qnn_path,
    }])
    qnn_stats.set_index('n_layers', inplace=True)

    return qnn_stats


def get_df_stats_fixed_layer(n_layer: int, qubits: int, seed: int, exp_id: int,
                             train_set, train_labels, test_set, test_labels, MODELS,
                             n_trials, n_jobs, n_train, n_test, dataset, tuning_bool,
                              use_stored_tuning, tuned_qubits_layer, tuned_global, noise_bool, noise_parameters,
                              realistic_gates, n_epochs, lr, debug_noise, metric_tuning, point_dimension, optimizer,
                              EXP_RESULTS_PATH, interface, regularization, noise_sources, save_qnn, save_data,
                              QNN_PATH,
                              tuning_dfs: dict[str, pd.DataFrame], tuned_qnn: dict[str, tuple[Optional[BaseQNN], float, float]]
                             ) -> dict:
    # Get optimal parameters
    if tuning_bool:
        print('Tuning started...')
        params = {
            model: tune_model(
                model=model,
                seed=seed,
                qubits=qubits,
                n_layers=n_layer,
                exp_id=exp_id,
                n_trials=n_trials,
                n_jobs=n_jobs,
                n_train=n_train,
                n_test=n_test,
                dataset=dataset,
                metric_tuning=metric_tuning,
                point_dimension=point_dimension,
                n_epochs=n_epochs,
                realistic_gates=realistic_gates,
                optimizer=optimizer,
                EXP_RESULTS_PATH=EXP_RESULTS_PATH,
                tuning_bool=tuning_bool,
                noise_bool=noise_bool,
                debug_noise=debug_noise,
                noise_parameters=noise_parameters,
                regularization=regularization,
                noise_sources=noise_sources,
                interface=interface,
                train_set=train_set,
                train_labels=train_labels,
                test_set=test_set,
                test_labels=test_labels,
                tuning_dfs=tuning_dfs,
                tuned_qnn=tuned_qnn
            )
            for model in MODELS
        }
        print('Tuning finished!')
    elif use_stored_tuning:
        params = {
            model: get_optimal_opt_parameters(tuned_qubits_layer, tuned_global, model, qubits, n_layer)
            for model in MODELS
        }
        print('Using stored optimal parameters from previuos tunings.')
    else:
        params = {
            model: {'lr': lr or DEFAULT_LRS[model]}
            for model in MODELS
        }
        print(f'Tuning is not scheduled. Using fixed learning rates {params}')

    # Use optimal parameters to ge stats
    df_stats = {}
    for model in MODELS:
        print(f'Evaluating model {model} performance')
        qnn = get_qnn(model=model, n_qubits=qubits, n_layers=n_layer, realistic_gates=realistic_gates, seed=seed,
                      interface=interface, noise=noise_bool, debug_noise=debug_noise, noise_parameters=noise_parameters,
                      regularization=regularization, noise_sources=noise_sources)
        df = train_and_evaluate(
            qnn=qnn,
            exp_id=exp_id,
            train_set=train_set,
            train_labels=train_labels,
            test_set=test_set,
            test_labels=test_labels,
            seed=seed,
            n_qubits=qubits,
            n_layers=n_layer,
            batch_size=BATCH_SIZE,
            lr=params[model]['lr'],
            n_epochs=n_epochs,
            save_qnn=save_qnn,
            optimizer=optimizer,
            dataset=dataset,
            realistic_gates=realistic_gates,
            save_data=save_data,
            QNN_PATH=QNN_PATH,
        )
        try:
            df = df.set_index('n_layers', drop=True)
        except Exception:
            pass
        df_stats[model] = df

        if save_qnn:
            qnn_path = get_qnn_path(model, qubits, n_layer, seed, exp_id, QNN_PATH)
            qnn.save_qnn(qnn_path)
            params_dict = {
                'n_qubits': qubits,
                'n_layers': n_layer,
                'lr': float(lr),
                'train_loss': float(df.iloc[-1]['train_loss']),
                'test_loss': float(df.iloc[-1]['test_loss']),
                'dataset': dataset,
                'metric': metric_tuning,
                'model': model,
                'point_dimension': point_dimension,
                'n_epochs': n_epochs,
                'batch_size': BATCH_SIZE,
                'optimizer': optimizer,
                'save_qnn': save_qnn,
                "realistic_gates": realistic_gates,
            }
            save_dict_to_json(params_dict, qnn_path.replace(f'{pickle_extension}', 'json'))

    return df_stats


################ SAVE RESULTS #################################
def get_exp_path(exp_id: int, EXP_RESULTS_PATH: str):
    path = os.path.join(EXP_RESULTS_PATH, f'results_{exp_id}.csv')
    return path


def get_qnn_path(model, n_qubits, n_layers, seed, exp_id, QNN_PATH: str):
    qnn_folder = f'{QNN_PATH.format(exp_id)}/{model}/exp_{exp_id}'
    qnn_path = os.path.join(qnn_folder, f'qnn_q_{n_qubits}_l_{n_layers}_s_{seed}.{pickle_extension}')
    return qnn_path


def get_data_path(dataset: str, model, exp_id, QNN_PATH: str) -> str:
    qnn_folder = f'{QNN_PATH.format(exp_id)}/{model}/exp_{exp_id}'
    data_path = os.path.join(qnn_folder, f'{dataset}.json')
    return data_path


def save_results_tuning(exp_id: int, qubits, n_layer: int, seed, study, config_dict: dict, model: str,
                        optimizer: str, EXP_RESULTS_PATH: str):

    path = get_exp_path(exp_id, EXP_RESULTS_PATH)
    path = path.replace('results_', f'tuning/{exp_id}/{model}_q_{qubits}_l_{n_layer}_s_{seed}')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save results
    results = [{"trial_number": trial.number,
                "params": trial.params,
                "cost": trial.value,
                'optimizer': optimizer}
               for trial in study.trials]
    df = pd.DataFrame(results).sort_values(by="cost", ascending=True)
    df.to_csv(path, index=False)
    print(f'Tuning results saved in {path}')

    # Save experiment configurations
    path_config = os.path.join(EXP_RESULTS_PATH, f'tuning//{exp_id}/config_tuning.json')
    save_dict_to_json(config_dict, path_config)


def save_results(df_stats: pd.DataFrame, exp_id: int, EXP_RESULTS_PATH):
    path = get_exp_path(exp_id, EXP_RESULTS_PATH)
    df_stats.to_csv(path)


def load_previous_results(experiment_folder: str, dataset: str, trained_models: list[str]) -> pd.DataFrame:
    metrics_model_cols = [f'{model}_train_loss' for model in trained_models]

    if os.path.exists(f'{experiment_folder}/all.csv'):
        all_df = pd.DataFrame(pd.read_csv(f'{experiment_folder}/all.csv'))
        # Filter dataset
        all_df = all_df[all_df['dataset'] == dataset]
    else:
        all_df = pd.DataFrame()

    # Read inner folders
    partial_dfs = []
    dataset_folder = f'{experiment_folder}/intermediate'
    dfs_dataset = []
    for folder in os.listdir(dataset_folder):
        if folder.startswith('exp_id') and os.path.isdir(f'{dataset_folder}/{folder}'):
            for f in os.listdir(f'{dataset_folder}/{folder}'):
                if not f.endswith('.csv'):
                    continue
                df = pd.DataFrame(pd.read_csv(f'{dataset_folder}/{folder}/{f}'))
                # Check that models are in df
                if any(metric not in df.columns for metric in metrics_model_cols):
                    continue
                dfs_dataset.append(df)

        # Concat all
        if not dfs_dataset:
            continue
        df_dataset = pd.concat(dfs_dataset, ignore_index=True)
        df_dataset['dataset'] = dataset

        # Drop null entries in metrics cols for trained_models
        df_dataset = df_dataset.dropna(subset=metrics_model_cols)

        partial_dfs.append(df_dataset)

    # Concat all datasets
    if not partial_dfs:
        df_partial = pd.DataFrame()
    else:
        df_partial = pd.concat(partial_dfs, ignore_index=True)

    # Concat with all
    df_all = pd.concat([all_df, df_partial], ignore_index=True)

    if df_all.empty:
        df_all = pd.DataFrame(columns=['n_qubits', 'n_layers', 'seed', 'dataset'])

    return df_all


def load_results_from_df(
    df: pd.DataFrame,
    MODELS: list[str],
    n_qubits: int,
    n_layers: int,
    seed: int
) -> pd.DataFrame:

    # Check models
    cols = ['n_qubits', 'n_layers', 'seed']
    for model in MODELS:
        cols += [f'{model}_train_loss', f'{model}_acc_train', f'{model}_acc_test', f'{model}_lr']

    # Filter by n_qubits, n_layers, seed
    df_experiment = df[
        (df['n_qubits'] == n_qubits) &
        (df['n_layers'] == n_layers) &
        (df['seed'] == seed)
    ]

    if len(df_experiment) != 0:
        return df_experiment[cols].copy()

    return pd.DataFrame()


def _create_paths(exp_id, MODEL_DIRS, QNN_PATH):
    for model, path in MODEL_DIRS.items():
        os.makedirs(os.path.join(QNN_PATH.format(exp_id), model), exist_ok=True)
        os.makedirs(path, exist_ok=True)


def main(args: Optional[dict] = None):
    # Read args
    if args is None:
        args = {}

    expected_args = set(DEFAULT_ARGS.keys())
    assert set(args.keys()).issubset(expected_args), f'Unexpected args: {set(args.keys()) - expected_args}'
    args = {**DEFAULT_ARGS, **args}

    realistic_gates = args['realistic_gates']
    n_qubits = args['n_qubits']
    layers_min = args['layers_min']
    layers_max = args['layers_max']
    layers_step = args['layers_step']
    regularization = args['regularization']
    metric_tuning = args['metric_tuning']
    trained_models = args['trained_models']
    n_trials = args['n_trials']
    n_jobs = args['n_jobs']
    n_train = args['n_train']
    n_test = args['n_test']
    n_epochs = args['n_epochs']
    point_dimension = args['point_dimension']
    n_seeds = args['n_seeds']
    starting_seed = args['starting_seed']
    dataset = args['dataset']
    optimizer = args['optimizer']
    interface = args['interface']
    LOAD_RESULTS = args['LOAD_RESULTS']
    save_qnn = args['save_qnn']
    save_data = args['save_data']
    debug_noise = args['debug_noise']
    tuning_bool = args['tuning_bool']
    eqk_bool = args['eqk_bool']
    noise_bool = args['noise_bool']
    use_stored_tuning = args['use_stored_tuning']
    lr = args['lr']
    noise_parameters_str = args['noise_parameters_str']
    noise_sources_str = args['noise_sources_str']
    experiment_folder = args['experiment_folder']

    if not tuning_bool and use_stored_tuning:
        tuned_qubits_layer, tuned_global = get_optimal_lr(optimizer, dataset)
    else:
        tuned_qubits_layer, tuned_global = None, None

    if n_qubits == 'all':
        n_qubits = [1, 2]
    elif isinstance(n_qubits, str) and n_qubits.isdigit():
        n_qubits = [int(n_qubits)]
    else:
        assert isinstance(n_qubits, int)
        n_qubits = [n_qubits]

    if eqk_bool:
        if not save_qnn:
            warn('If eqk is True, save_qnn must be True. It has been changed', category=UserWarning, stacklevel=2)
        save_qnn = True

    empty_df = pd.DataFrame([], columns=['n_qubits', 'n_layers', 'seed', 'train_loss', 'test_loss',
                                         'acc_train', 'acc_test', 'lr'])

    ################ Process noise options ################
    noise_parameters = ast.literal_eval(noise_parameters_str)
    noise_sources = []
    if 'depolarizing' in noise_sources_str or noise_sources_str == 'all':
        noise_sources.append('depolarizing')
    if 'amplitude' in noise_sources_str or noise_sources_str == 'all':
        noise_sources.append('amplitude')
    if 'phase' in noise_sources_str or noise_sources_str == 'all':
        noise_sources.append('phase')

    ################ EXPERIMENT PATHS #################################
    suffix_noise = '_noise' if noise_bool else ''
    suffix_debug = '_debug' if debug_noise else ''
    suffix = f'{suffix_noise}{suffix_debug}'
    device_folder = _get_device_folder(noise_parameters, noise_sources)
    if experiment_folder is None:
        experiment_folder = f'final_experiment{suffix}'
    EXP_RESULTS_PATH = os.path.join(get_root_path('Pulsed-Data-Reuploading-Quantum-Models'),
                                    f'data/results/{experiment_folder}/{device_folder}/{dataset}')
    QNN_PATH = os.path.join(EXP_RESULTS_PATH, 'trained_qnn/')
    os.makedirs(EXP_RESULTS_PATH, exist_ok=True)
    if tuning_bool:
        os.makedirs(f'{EXP_RESULTS_PATH}/tuning', exist_ok=True)
    INTERMEDIATE_FOLDER = os.path.join(EXP_RESULTS_PATH, f'intermediate')
    os.makedirs(INTERMEDIATE_FOLDER, exist_ok=True)
    ic(EXP_RESULTS_PATH, INTERMEDIATE_FOLDER)

    start_experiment = time()
    exp_id = get_highest_id(EXP_RESULTS_PATH) + 1
    base_dir_exp = os.path.join(INTERMEDIATE_FOLDER, f'exp_id_{exp_id}')

    MODEL_PATHS = {
        name: os.path.join(base_dir_exp, name)
        for name in ['gate', 'gate_spherical', 'mixed', 'pulsed']
    }

    if trained_models == 'all':
        MODELS = list(MODEL_PATHS.keys())
        MODEL_DIRS = MODEL_PATHS
    else:
        MODELS = [m for m in MODEL_PATHS if m in trained_models]
        MODEL_DIRS = {m: MODEL_PATHS[m] for m in MODELS}

    _create_paths(exp_id, MODEL_DIRS, QNN_PATH)

    df_stats_partial = pd.DataFrame()
    df_prev_resuls = load_previous_results(EXP_RESULTS_PATH, dataset, MODELS)
    # Tuning experiment
    range_layers = range(layers_min, layers_max + 1, layers_step)
    for n_qubit in n_qubits:
        for n_layer in range_layers:
            if n_layer == 0:
                n_layer = 1
            # Initialize dataframes
            model_intermediates = {model: pd.DataFrame() for model in MODELS}
            if eqk_bool:
                df_eqk_intermediate = pd.DataFrame()

            for seed in range(starting_seed, starting_seed + n_seeds):
                
                if tuning_bool:
                    tuning_dfs = {model: empty_df.copy() for model in MODELS}
                    tuned_qnn: dict[str, tuple[Optional[BaseQNN], float, float]] = (
                        {model: (None, float('inf'), float('inf')) for model in MODELS}
                    )
                # Prepare intermediate paths
                model_paths = {
                    model: os.path.join(MODEL_DIRS[model],
                                        f'qubits={n_qubit}layers={n_layer}_seed_{seed}_exp_id_{exp_id}.csv')
                    for model in MODELS
                }

                # Get Dataset
                train_set, train_labels, test_set, test_labels = get_dataset(
                    dataset,
                    n_train,
                    n_test,
                    interface,
                    points_dimension=point_dimension,
                    seed=seed
                )

                # Check for existing results
                print('---------')
                save_results_ = False
                all_loaded = False
                if LOAD_RESULTS:
                    stats_dict = {}
                    df_loaded_exp = load_results_from_df(
                            df_prev_resuls, MODELS, n_qubit, n_layer, seed
                    )

                    model_metric_cols = [
                        f'{model}_{c}' for c in [
                            'train_loss',
                            'acc_train',
                            'acc_test',
                            'lr'] for model in MODELS]
                    if set(model_metric_cols).issubset(df_loaded_exp.columns):
                        for model in MODELS:
                            df_model = df_loaded_exp[
                                ['n_qubits', 'n_layers', 'seed',
                                 f'{model}_train_loss', f'{model}_acc_train', f'{model}_acc_test', f'{model}_lr']
                            ].copy()
                            df_model.set_index(['n_layers'], inplace=True)
                            if len(df_model) == 0:
                                break
                            stats_dict[model] = df_model.iloc[[0], :]
                        if len(stats_dict) == len(MODELS):
                            all_loaded = True

                if not all_loaded:
                    print_in_blue(f"[INFO] Running experiment for QUBITS: {n_qubit} - LAYERS: {n_layer} - SEED {seed}")
                    stats_dict = get_df_stats_fixed_layer(
                        n_layer=n_layer,
                        qubits=n_qubit,
                        seed=seed,
                        exp_id=exp_id,
                        train_set=train_set,
                        train_labels=train_labels,
                        test_set=test_set,
                        test_labels=test_labels,
                        MODELS=MODELS,
                        n_trials=n_trials,
                        n_jobs=n_jobs,
                        n_train=n_train,
                        n_test=n_test,
                        dataset=dataset,
                        tuning_bool=tuning_bool,
                        use_stored_tuning=use_stored_tuning,
                        tuned_qubits_layer=tuned_qubits_layer,
                        tuned_global=tuned_global,
                        noise_bool=noise_bool,
                        noise_parameters=noise_parameters,
                        realistic_gates=realistic_gates,
                        n_epochs=n_epochs,
                        lr=lr,
                        debug_noise=debug_noise,
                        metric_tuning=metric_tuning,
                        point_dimension=point_dimension,
                        optimizer=optimizer,
                        EXP_RESULTS_PATH=EXP_RESULTS_PATH,
                        interface=interface,
                        regularization=regularization,
                        noise_sources=noise_sources,
                        save_qnn=save_qnn,
                        save_data=save_data,
                        QNN_PATH=QNN_PATH,
                        tuning_dfs=tuning_dfs,
                        tuned_qnn=tuned_qnn,
                    )
                    save_results_ = True
                else:
                    print_in_blue(
                        f"[INFO] Loaded previous results for QUBITS: {n_qubit} - LAYERS: {n_layer} - SEED {seed}")
                print('---------')

                # Get the model results to save them separately
                for model in MODELS:
                    df = stats_dict[model]
                    df['n_layers'] = df.copy().index
                    df.set_index(['n_qubits', 'n_layers', 'seed'], inplace=True, drop=True)
                    model_intermediates[model] = pd.concat([model_intermediates[model], df], axis=0)
                    if save_results_:
                        for model, path in model_paths.items():
                            df = stats_dict[model]

            # Merge stats
            for model in MODELS:
                df = model_intermediates[model]
                drop_cols = set(df.columns) - {'train_loss', 'test_loss', 'acc_train', 'acc_test', 'lr', 'seed'}
                df.drop(columns=drop_cols, inplace=True, errors='ignore')
                df.columns = [f"{model}_{col}" for col in df.columns]

            dfs_to_concat = [model_intermediates[model] for model in MODELS]

            # Get the summary stats for seed, dataset, n_qubits and layers
            df_partial = pd.concat(dfs_to_concat, axis=1)

            # Join the stats for seed, dataset, n_qubits and layers in the merged dataframe
            df_stats_partial = pd.concat([df_stats_partial, df_partial], axis=0)

            # save intermediate results
            if not all_loaded:
                layer_path_df = os.path.join(INTERMEDIATE_FOLDER,
                                             f"exp_id_{exp_id}/qubit_{n_qubit}_layer_{n_layer}.csv")
                df_partial.to_csv(layer_path_df)
                print(
                    f'[INFO] Saved intermediate results for QUBITS: {n_qubit} - LAYERS: {n_layer} - SEED {seed} in {layer_path_df}')

                # config layer to json
                config_layer_path_json = os.path.join(INTERMEDIATE_FOLDER,
                                                        f"exp_id_{exp_id}/qubit_{n_qubit}_layer_{n_layer}.json")
                config_layer = {
                    'n_qubits': n_qubit,
                    'layers_min': layers_min,
                    'layers_max': layers_max,
                    'n_trials': n_trials,
                    'n_jobs': n_jobs,
                    'n_train': n_train,
                    'n_test': n_test,
                    'point_dimension': point_dimension,
                    'n_seeds': n_seeds,
                    'dataset': dataset,
                    'optimizer': optimizer,
                    'interface': interface,
                    'LOAD_RESULTS': LOAD_RESULTS,
                    'tuning_bool': tuning_bool,
                    'lr': float(lr),
                    'n_epochs': n_epochs,
                    "realistic_gates": realistic_gates,
                }
                save_dict_to_json(config_layer, config_layer_path_json)

    # Save experiment configuration and results
    path = get_exp_path(exp_id, EXP_RESULTS_PATH)
    df_stats_partial.to_csv(path)
    spent_time = time() - start_experiment
    time_per_experiment = spent_time / (len(df_stats_partial) or 1)
    config_layer = {
        'n_qubits': n_qubit, 'layers_min': layers_min, 'layers_max': layers_max, 'n_trials': n_trials,
        'n_jobs': n_jobs, 'n_train': n_train, 'n_test': n_test, 'point_dimension': point_dimension,
        'n_seeds': n_seeds, 'dataset': dataset, 'optimizer': optimizer, 'interface': interface, 'n_epochs': n_epochs,
        'LOAD_RESULTS': LOAD_RESULTS, 'tuning_bool': tuning_bool, 'lr': float(lr),
        'spent_time': spent_time, 'time_per_experiment': time_per_experiment,
    }
    save_dict_to_json(config_layer, path.replace('results_', 'config_').replace('.csv', '.json'))

    print_in_blue('\n\n[END] Experiment finished!')
    print_in_blue(f'[INFO] Total time spent:         {spent_time:.2f} seconds')
    print_in_blue(f'[INFO] Mean time per experiment: {time_per_experiment:.2f} seconds')
    print_in_blue(f'[INFO] Experiment results saved to {path}')


################ MAIN #################################
if __name__ == '__main__':
    main(args=None)

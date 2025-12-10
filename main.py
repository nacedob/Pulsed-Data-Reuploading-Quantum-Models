import subprocess
from icecream import ic
from warnings import filterwarnings
from src.utils import load_json_to_dict
import numpy as np

noise_mapper = load_json_to_dict('data/backends/custom_mapper.json')
low_noise_device = str(noise_mapper['low_noise']).replace(' ', '')
middle_noise_device = str(noise_mapper['middle_noise']).replace(' ', '')
high_noise_device = str(noise_mapper['high_noise']).replace(' ', '')

filterwarnings('ignore', category=RuntimeWarning)
filterwarnings('ignore', category=Warning)

JOBS = 4

MODELS = "['gate','mixed']"   # 'loss'
DEBUG_NOISE = False
LOAD = True
REGULARIZATION_QNN = 0
REGULARIZATION_TUNING = 0
METRIC_TUNING = 'loss'   # 'loss'
n_points_train = 300
n_points_test = 100

common_str = ('python -m src.experiments.final_experiment --n_seeds={seeds}  --layers_max={layers_max} '
              '--layers_min={layers_min} --tuning=True --dataset={dataset} --noise={noise} --layers_step={layers_step} '
              '--n_epochs=30 --trials=30 --optimizer=rms --n_jobs={n_jobs} --n_qubits={qubits} '
              '--realistic_gates=False --save_qnn=True --eqk=False --starting_seed={starting_seed} '
              '--trained_models={models} --noise_parameters={noise_parameters} --metric_tuning={metric} '
              f'--n_train={n_points_train} --n_test={n_points_test} '
              f'--regularization={REGULARIZATION_QNN} --debug_noise={DEBUG_NOISE} --load={LOAD}')


noise_parameters = np.arange(0, 0.3001, 0.03).round(3)
NOISE_SOURCES = ['all']
datasets = ['digits_08']   #, 'fashion', 'shell', 'digits_17' , 'helix', 'iris']


def define_experiment_str(seed: int):
    """
    Esta funcion la he hecho simplemente para poder hacer que los experimentos se corran primero todos aquellos de la 
    seed 0 (para todos los datasets y modelos), despues para el seed 1...
    Si metes mas de una seed a la vez, entonces priorizas resolver un dataset entero variando seeds. 
    Asi podemos ir viendo los resultados y la forma que tiene de una manera distinta
    """
    commands = []
    for dataset in datasets:
        for p in noise_parameters:
            p = float(p)
            # Fix n_kayers to 20 and noise parameters are changing
            commands += [common_str.format(
                dataset=dataset,
                qubits=2,
                layers_min=20,
                layers_max=21,
                layers_step=5,
                noise=True,
                seeds=1,
                starting_seed=seed,
                n_jobs=JOBS,
                models=MODELS,
                noise_parameters=str({'depolarizing_1q': p}).replace(' ', ''),
                metric=METRIC_TUNING)
            ]
    return commands

experiment_str = []
for seed in range(6):
    experiment_str += define_experiment_str(seed)


for cmd in experiment_str:
    ic(cmd)
    subprocess.run(cmd, shell=True, text=True, capture_output=False, check=True)
    ic(f'[INFO CMD] FINISHED')

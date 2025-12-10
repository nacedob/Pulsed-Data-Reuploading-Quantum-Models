from icecream import ic
from warnings import filterwarnings
from src.experiments.final_experiment import main
from ast import literal_eval
import sys

if len(sys.argv) > 1:
    model_list = eval(sys.argv[1])
else:
    model_list = ['gate','mixed']

if len(sys.argv) > 2:
    seed_list = literal_eval(sys.argv[2])
else:
    seed_list = range(5)

filterwarnings('ignore', category=RuntimeWarning)
filterwarnings('ignore', category=Warning)

JOBS = 5

MODELS = "['gate','mixed']"   # 'loss'
DEBUG_NOISE = False
REGULARIZATION_QNN = 0
REGULARIZATION_TUNING = 0
METRIC_TUNING = 'loss'   # 'loss'
n_points_train = 300
n_points_test = 100
EXPERIMENT_FOLDER = 'LAYERS_FOLDER'

NOISE_SOURCES = ['all']
datasets = ['digits_08']   #, 'digits_17', 'fashion', 'shell', 'helix']


args = {
    'n_qubits': 2,
    'n_seeds': 1 ,
    'layers_min': 0,
    'layers_max': 50,
    'layers_step': 5,
    'n_epochs': 30,  
    'n_trials': 30,
    'n_jobs': JOBS,
    'tuning_bool': True,
    'noise_bool': True,
    'LOAD_RESULTS': True,
    'optimizer': 'rms',
    'realistic_gates': False,
    'save_qnn': True,
    'eqk_bool': False,
    'trained_models': model_list,
    'metric_tuning': 'loss',
    'n_train': n_points_train,
    'n_test': n_points_test,
    'experiment_folder': EXPERIMENT_FOLDER,
    'regularization': 0,
    'debug_noise': DEBUG_NOISE
    }

def get_args(dataset: str, seed: int) -> dict:
    args_ = args.copy()
    args_['dataset'] = dataset
    args_['starting_seed'] = seed
    return args_


for dataset in datasets:
    for seed in seed_list:
        ic(dataset, seed)
        main(get_args(dataset, seed))
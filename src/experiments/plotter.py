from icecream import ic
from config import get_root_path
import pandas as pd
from pathlib import Path
import numpy as np
from src.experiments.visualization.final_figure_layer_and_noise import main


ROOT = Path(get_root_path())
RESULTS_FOLDER = ROOT / 'data' / 'results'
PREVIOUS_FOLDER = RESULTS_FOLDER / 'previous'
NOISE_FOLDER = RESULTS_FOLDER / 'NOISE_EXPERIMENT'
LAYER_FOLDER = RESULTS_FOLDER / 'LAYERS_EXPERIMENT'
RESULT_FOLDER = ROOT / 'data' / 'results' 
SAVE_FOLDER = RESULT_FOLDER / 'figures' 
METRIC = 'accuracy'


def build_block(df, model):
    return pd.DataFrame({
        "n_qubits": df["n_qubits"],
        "n_layers": df["n_layers"],
        "seed": df["seed"],
        "dataset": df["dataset"],
        "model": model,
        "p": df["p"] if "p" in df.columns else None,
        "train_loss": df[f"{model}_train_loss"],
        "test_loss": np.nan,  # not present in your CSV
        "acc_train": df[f"{model}_acc_train"],
        "acc_test": df[f"{model}_acc_test"],
        "lr": df[f"{model}_lr"],
    })


def format_previous_noise(previous_path: Path) -> pd.DataFrame:
    df = pd.read_csv(previous_path)
    out = pd.concat(
        [build_block(df, "gate"), build_block(df, "mixed")],
        ignore_index=True
    )
    return out


def merge_noise():
    dfs = []
    # Read previous results
    previous_noise = format_previous_noise(PREVIOUS_FOLDER / 'all_noise_digits_08.csv')
    dfs.append(previous_noise)

    # Read new results
    for device in NOISE_FOLDER.iterdir():
        p = float(device.name.split('depolarizing_1q_')[1])

        for dataset in device.iterdir():
            for result in dataset.iterdir():
                # Finished experiments (e.g. results_0.csv)
                if result.name.endswith('.csv') and result.name.startswith('results'):
                    df = pd.read_csv(result)
                    df['p'] = p
                    dfs.append(df)
                # partial folder
                if not result.name == 'partial':
                    continue
                
                for partial_result in result.iterdir():
                    if not partial_result.name.endswith('.csv'):
                        continue
                    df = pd.read_csv(partial_result)
                    dfs.append(df)
                    df['p'] = p

    # merge
    df = pd.concat(dfs, ignore_index=True)

    return df


def merge_layers():
    dfs = []

    # Read previous results
    previous_layers = format_previous_noise(PREVIOUS_FOLDER / 'all_layers_digits_08.csv')
    dfs.append(previous_layers)

    # Read new results
    for dataset in (LAYER_FOLDER / 'default').iterdir():
        for result in dataset.iterdir():
            # Finished experiments (e.g. results_0.csv)
            if result.name.endswith('.csv') and result.name.startswith('results'):
                df = pd.read_csv(result)
                dfs.append(df)

            # partial folder
            if not result.name == 'partial':
                continue
            
            for partial_result in result.iterdir():
                if not partial_result.name.endswith('.csv'):
                    continue
                df = pd.read_csv(partial_result)
                dfs.append(df)
            

    # merge
    df = pd.concat(dfs, ignore_index=True)
    return df




if __name__ == '__main__':
    noise_df = merge_noise()
    layers_df = merge_layers()
    dataset_list = set(noise_df["dataset"].unique()) | set(layers_df["dataset"].unique())
    
    final_result_folder = RESULT_FOLDER / 'merged'
    final_result_folder.mkdir(parents=True, exist_ok=True)
    noise_df.to_csv(final_result_folder / 'noise.csv', index=False)
    layers_df.to_csv(final_result_folder / 'layers.csv', index=False)
    
    # Filter till 40 layers
    layers_df = layers_df[layers_df["n_layers"] <= 40].copy()
        
    for dataset in dataset_list:
        main(dataset=dataset, metric=METRIC, df_layers=layers_df, df_noise=noise_df, save_folder=SAVE_FOLDER)

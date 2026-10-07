import os
import pandas as pd
from src.experiments.visualization.final_figure_layer_and_noise import dataset_mapping


def layers_table(csv_path: str) -> str:
    # 1. Load the data
    df = pd.read_csv(csv_path)
    target_layers = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    target_datasets = ["digits_08", "iris", "corners3d", "helix"]
    df = df[
        df["dataset"].isin(target_datasets) & df["n_layers"].isin(target_layers)
    ]

    # Map the dataset column to its clean presentation string
    if isinstance(dataset_mapping, dict):
        df["dataset"] = df["dataset"].map(dataset_mapping)
    else:
        df["dataset"] = df["dataset"].apply(dataset_mapping)

    # 2. Compute performance metrics (mean and standard deviation) across seeds
    grouped = (
        df.groupby(["dataset", "n_qubits", "n_layers", "model"])[
            ["acc_train", "acc_test"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "dataset",
        "n_qubits",
        "n_layers",
        "model",
        "train_mean",
        "train_std",
        "test_mean",
        "test_std",
    ]

    # 3. Separate and merge models side by side
    gate_df = grouped[grouped["model"] == "gate"]
    mixed_df = grouped[grouped["model"] == "mixed"]
    merged = pd.merge(
        gate_df,
        mixed_df,
        on=["dataset", "n_qubits", "n_layers"],
        suffixes=("_gate", "_mixed"),
    )

    # Sort for structural order
    merged = merged.sort_values(["dataset", "n_layers"])

    # 4. Construct LaTeX rows
    latex_rows = []
    datasets = merged["dataset"].unique()

    for i, ds in enumerate(datasets):
        # Filter for the current mapped dataset name
        ds_df = merged[merged["dataset"] == ds]

        for idx, row in enumerate(ds_df.itertuples()):
            # Escape underscore if any remaining raw characters exist
            ds_name = row.dataset.replace("_", "\\_") if idx == 0 else ""

            # Format numbers as Mean \pm Std strictly within math mode $...$
            g_tr = (
                f"${row.train_mean_gate:.3f} \\pm {row.train_std_gate:.3f}$"
                if not pd.isna(row.train_std_gate)
                else f"${row.train_mean_gate:.3f}$"
            )
            g_te = (
                f"${row.test_mean_gate:.3f} \\pm {row.test_std_gate:.3f}$"
                if not pd.isna(row.test_std_gate)
                else f"${row.test_mean_gate:.3f}$"
            )
            m_tr = (
                f"${row.train_mean_mixed:.3f} \\pm {row.train_std_mixed:.3f}$"
                if not pd.isna(row.train_std_mixed)
                else f"${row.train_mean_mixed:.3f}$"
            )
            m_te = (
                f"${row.test_mean_mixed:.3f} \\pm {row.test_std_mixed:.3f}$"
                if not pd.isna(row.test_std_mixed)
                else f"${row.test_mean_mixed:.3f}$"
            )

            line = f"{ds_name} & {row.n_qubits} & {row.n_layers} & {g_tr} & {g_te} & & {m_tr} & {m_te} \\\\"
            latex_rows.append(line)

        # Draw a line between dataset blocks (omitted after the last block)
        if i < len(datasets) - 1:
            latex_rows.append("\\cline{2-8}")

    # Assemble the final template string with table* environment and bottom caption
    table_template = (
        f"""\\begin{{table*}}[ht]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{12pt}}
\\begin{{tabular}}{{cccccccc}}
\\hline
 & & & \\multicolumn{{2}}{{c}}{{\\textbf{{Gate-Based Model}}}} & & \\multicolumn{{2}}{{c}}{{\\textbf{{Proposed Pulsed Model}}}} \\\\ \\cline{{4-5}} \\cline{{7-8}} 
\\textbf{{Dataset}} & \\textbf{{Qubits}} & \\textbf{{Layers}} & $\\mathbf{{\\text{{Acc}}_{{\\text{{train}}}}}}$ & $\\mathbf{{\\text{{Acc}}_{{\\text{{test}}}}}}$ & & $\\mathbf{{\\text{{Acc}}_{{\\text{{train}}}}}}$ & $\\mathbf{{\\text{{Acc}}_{{\\text{{test}}}}}}$ \\\\ \\hline
"""
        + "\n".join(latex_rows)
        + """
\\hline
\\end{tabular}
\\caption{Comparative performance analysis between the conventional gate-based model and the proposed pulsed model across varying data distributions and layer counts. Statistical metrics denote the $\\text{mean} \\pm \\text{standard deviation}$ calculated over all evaluation trials.}
\\label{tab:results_layers}
\\end{table*}
"""
    )

    return table_template


def noise_table(csv_path: str) -> str:
    # 1. Load the data
    df = pd.read_csv(csv_path)
    
    # Filter for the fixed layer slice (L=20) and target datasets
    # Note: Adjust your target_p list based on the precise noise rates present in your noise CSV file
    target_datasets = ["digits_08", "iris", "corners3d", "helix"]
    
    df = df[
        (df["n_layers"] == 20) & 
        (df["dataset"].isin(target_datasets)) & 
        (df["p"] < 0.3 if "p" in df.columns and df["p"].notna().any() else True)
    ]

    # Map the dataset column to its clean presentation string
    if isinstance(dataset_mapping, dict):
        df["dataset"] = df["dataset"].map(dataset_mapping)
    else:
        df["dataset"] = df["dataset"].apply(dataset_mapping)

    # 2. Compute performance metrics (mean and standard deviation) across seeds
    # Grouping by 'p' instead of 'n_layers' for this experiment
    grouped = (
        df.groupby(["dataset", "n_qubits", "p", "model"])[
            ["acc_train", "acc_test"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "dataset",
        "n_qubits",
        "p",
        "model",
        "train_mean",
        "train_std",
        "test_mean",
        "test_std",
    ]

    # 3. Separate and merge models side by side
    gate_df = grouped[grouped["model"] == "gate"]
    mixed_df = grouped[grouped["model"] == "mixed"]
    merged = pd.merge(
        gate_df,
        mixed_df,
        on=["dataset", "n_qubits", "p"],
        suffixes=("_gate", "_mixed"),
    )

    # Sort structurally by dataset and noise rate
    merged = merged.sort_values(["dataset", "p"])

    # 4. Construct LaTeX rows
    latex_rows = []
    datasets = merged["dataset"].unique()

    for i, ds in enumerate(datasets):
        # Filter for the current mapped dataset name
        ds_df = merged[merged["dataset"] == ds]

        for idx, row in enumerate(ds_df.itertuples()):
            # Only display dataset name on the first row of its block
            ds_name = row.dataset.replace("_", "\\_") if idx == 0 else ""

            # Format numbers as Mean \pm Std strictly within math mode $...$
            g_tr = (
                f"${row.train_mean_gate:.3f} \\pm {row.train_std_gate:.3f}$"
                if not pd.isna(row.train_std_gate)
                else f"${row.train_mean_gate:.3f}$"
            )
            g_te = (
                f"${row.test_mean_gate:.3f} \\pm {row.test_std_gate:.3f}$"
                if not pd.isna(row.test_std_gate)
                else f"${row.test_mean_gate:.3f}$"
            )
            m_tr = (
                f"${row.train_mean_mixed:.3f} \\pm {row.train_std_mixed:.3f}$"
                if not pd.isna(row.train_std_mixed)
                else f"${row.train_mean_mixed:.3f}$"
            )
            m_te = (
                f"${row.test_mean_mixed:.3f} \\pm {row.test_std_mixed:.3f}$"
                if not pd.isna(row.test_std_mixed)
                else f"${row.test_mean_mixed:.3f}$"
            )

            # Render noise column formatted nicely to 1 decimal place or as raw float
            p_val = f"{row.p:.2f}" if isinstance(row.p, (int, float)) else str(row.p)

            line = f"{ds_name} & {row.n_qubits} & {p_val} & {g_tr} & {g_te} & & {m_tr} & {m_te} \\\\"
            latex_rows.append(line)

        # Draw a line between dataset blocks (omitted after the last block)
        if i < len(datasets) - 1:
            latex_rows.append("\\cline{2-8}")

    # Assemble the final template string matching the layout of the layers script
    table_template = (
        f"""\\begin{{table*}}[ht]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{12pt}}
\\begin{{tabular}}{{cccccccc}}
\\hline
 & & & \\multicolumn{{2}}{{c}}{{\\textbf{{Gate-Based Model}}}} & & \\multicolumn{{2}}{{c}}{{\\textbf{{Proposed Pulsed Model}}}} \\\\ \\cline{{4-5}} \\cline{{7-8}} 
\\textbf{{Dataset}} & \\textbf{{Qubits}} & \\textbf{{Noise}} & $\\mathbf{{\\text{{Acc}}_{{\\text{{train}}}}}}$ & $\\mathbf{{\\text{{Acc}}_{{\\text{{test}}}}}}$ & & $\\mathbf{{\\text{{Acc}}_{{\\text{{train}}}}}}$ & $\\mathbf{{\\text{{Acc}}_{{\\text{{test}}}}}}$ \\\\ \\hline
"""
        + "\n".join(latex_rows)
        + """
\\hline
\\end{tabular}
\\caption{Model robustness and evaluation accuracy under varying levels of depolarizing channel noise ($p$) with depth fixed at $L=20$ layers. Statistical variations reflect the $\\text{mean} \\pm \\text{standard deviation}$ calculated across all independent execution seeds.}
\\label{tab:results_noise}
\\end{table*}
"""
    )

    return table_template

# --- Execute and display ---
if __name__ == "__main__":
    folder = "data/results/merged"
    
    # Layers table
    csv_filename = f"{folder}/layers.csv"
    summary_table = layers_table(csv_filename)
    with open(f"{folder}/layers_table.tex", "w", encoding="utf-8") as f:
        f.write(summary_table)
        print(f"Successfully wrote table to {folder}/layers_table.tex")
    
    # Noise table
    csv_filename = f"{folder}/noise.csv"
    summary_table = noise_table(csv_filename)
    with open(f"{folder}/noise_table.tex", "w", encoding="utf-8") as f:
        f.write(summary_table)
        print(f"Successfully wrote table to {folder}/noise_table.tex")
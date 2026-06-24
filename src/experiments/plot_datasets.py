from src.experiments.config_exp import get_dataset
from src.experiments.visualization.utils import styles
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


SAVE_FOLDER = Path('data/results/figures/datasets')
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

def plot_dataset(dataset: str, elev=20, azim=45, s=20):
    
    color_1 = styles['gate']['color']
    color_0 = styles['mixed']['color']
    
    X_train, y_train, _, _ = get_dataset(
        dataset=dataset, n_train=3000, n_test=3, points_dimension=3, seed=0, interface='jax'
    )
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    assert X.shape[1] == 3, "X must be of shape (N, 3)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Two classes
    mask0 = (y == 0)
    mask1 = (y == 1)

    ax.scatter(X[mask0, 0], X[mask0, 1], X[mask0, 2],
               c=color_0, s=s, label='Class 0', alpha=0.6)

    ax.scatter(X[mask1, 0], X[mask1, 1], X[mask1, 2],
               c=color_1, s=s, label='Class 1', alpha=0.6)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    
    # extra safety (removes 3D panes/background)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color((0, 0, 0, 0))
    
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(SAVE_FOLDER / f'{dataset}.png', dpi=300, transparent=True)
    print(f'Saved {dataset} plot to {SAVE_FOLDER / f"{dataset}.png"}')
    
    

def plot_legend():
    color_1 = styles['gate']['color']
    color_0 = styles['mixed']['color']
    
    # 1. Definimos un tamaño de figura más pequeño y alargado, ideal para una leyenda
    fig = plt.figure(figsize=(3, 1.5)) 
    ax = fig.add_subplot(111)
    
    # Creamos los elementos invisibles para la leyenda
    ax.scatter([], [], c=color_0, label='Class 0')
    ax.scatter([], [], c=color_1, label='Class 1')
    
    # 2. Ocultamos los ejes por completo para que no se vea nada del gráfico
    ax.axis('off')
    
    # 3. Dibujamos la leyenda centrada y con una fuente más grande (propiedad fontsize)
    # 'loc=10' o 'loc="center"' la ubica justo en medio de la figura
    ax.legend(loc='center', fontsize=14, ncol=2, frameon=True)
    
    # 4. Ajustamos los márgenes para que el recuadro no se corte y guardamos
    plt.tight_layout()
    plt.savefig(SAVE_FOLDER / 'legend.png', dpi=300, transparent=True)
    print(f'Saved legend plot to {SAVE_FOLDER / "legend.png"}')
    
    
if __name__ == '__main__':
    plot_legend()
    exit()
    plot_dataset('corners3d')
    plot_dataset('helix')
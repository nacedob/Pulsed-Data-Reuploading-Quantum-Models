from typing import Any

STYLES: dict[str, dict[str, Any]] = {
    "gate": {
        "model": "gate",
        "color": "#6C3C84",
        "marker": "s",
        "name": "Gate",
    },
    "mixed": {
        "model": "mixed",
        "color": "#D435C4",
        "marker": "o",
        "name": "Pulsed",
    },
    "gate_spherical": {
        "model": "spherical_gate",
        "color": "#2ca02c",
        "marker": "^",
        "name": "Gate spherical",
    },
    "pulsed": {
        "model": "pulsed",
        "color": "#6baed6",
        "marker": "D",
        "name": "Fully pulsed",
    },
}


DATASET_NAMES = {
    "digits_08": "MNIST Digits",
    "helix": "Helix",
    "iris": "Iris Flowers",
}

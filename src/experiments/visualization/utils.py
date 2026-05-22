from matplotlib.colors import to_rgb, to_hex
import numpy as np
from colour import Color

def darken_color(color, factor=0.7):
    """Oscurece un color en formato hex al reducir su brillo."""
    rgb = np.array(to_rgb(color))
    darkened_rgb = np.clip(rgb * factor, 0, 1)
    return to_hex(darkened_rgb)


def lighten_color(color, factor=1.3):
    """Oscurece un color en formato hex al reducir su brillo."""
    rgb = np.array(to_rgb(color))
    darkened_rgb = np.clip(rgb * factor, 0, 1)
    return to_hex(darkened_rgb)


def create_color_gradient(color1, color2, n_points):
    """
    Create a smooth gradient of colors between two specified colors.

    Parameters:
    - color1 (str or tuple): The starting color (e.g., 'red', '#FF0000', or RGB tuple).
    - color2 (str or tuple): The ending color (e.g., 'blue', '#0000FF', or RGB tuple).
    - n_points (int): Number of points (colors) in the gradient.

    Returns:
    - gradient (list): List of RGB tuples forming the color gradient.
    """
    raise NotImplementedError
    c1 = Color(color1)
    c2 = Color(color2)
    gradient = [to_rgb(c.hex) for c in c1.range_to(c2, n_points)]
    return gradient

styles = {
        'gate': {'model': 'gate', 'color': '#6C3C84', 'marker': 's', 'name': 'Gate'},
        'mixed': {'model': 'mixed', 'color': '#D435C4', 'marker': 'o', 'name': 'Pulsed'},
        'gate_spherical': {'model': 'spherical_gate', 'color': '#2ca02c', 'marker': '^', 'name': 'Gate spherical'},
        'pulsed': {'model': 'pulsed', 'color': '#6baed6', 'marker': 'D', 'name': 'Fully pulsed'}
}

dataset_names = {
    'digits_08': 'MNIST Digits',
    'helix': 'Helix',
    'iris': 'Iris Flowers',
}

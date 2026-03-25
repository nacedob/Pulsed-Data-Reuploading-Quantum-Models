# Pulsed Data Re-Uploading Quantum Models

This repository contains numerical experiments for the paper **"Pulsed Learning for Quantum Data Re-Uploading Models"**: [arXiv:2512.10670](https://arxiv.org/abs/2512.10670).


In this work, we introduce a **pulse-based variant of data re-uploading**, benchmarked on a simulated superconducting transmon processor with realistic noise. Our results demonstrate that pulsed models:

- Outperform gate-based counterparts in test accuracy and generalization under noise.
- Maintain higher fidelity under increasing noise, showing enhanced resilience to decoherence and control errors.
- Highlight the potential of pulse-native architectures as a practical, hardware-aligned path for QML on NISQ devices.

---
## Abstract
While Quantum Machine Learning (QML) holds great potential, its practical realization on Noisy Intermediate-Scale Quantum (NISQ) hardware has been hindered by the limitations of variational quantum circuits (VQCs). Recent evidence suggests that VQCs suffer from severe trainability and noise-related issues, leading to growing skepticism about their long-term viability. However, the possibility of implementing learning models directly at the pulse-control level remains comparatively unexplored and could offer a promising alternative. In this work, we formulate a pulse-based variant of data re-uploading, embedding trainable parameters directly into the native system's dynamics. We benchmark our approach on a simulated superconducting transmon processor with realistic noise profiles. The pulse-based model consistently outperforms its gate-based counterpart, exhibiting higher test accuracy and improved generalization under equivalent noise conditions. Moreover, by systematically increasing noise strength, we show that pulse-level implementations retain higher fidelity for longer, demonstrating enhanced resilience to decoherence and control errors. These results suggest that pulse-native architectures, though less explored, may offer a viable and hardware-aligned path forward for practical QML in the NISQ era. 

---

## Features

- **Pulsed QML Framework**  
  Train quantum neural networks using parameterized pulse controls instead of traditional gate sequences.

- **Gate vs. Pulse Comparisons**  
  Benchmark pulsed, gate-based, and hybrid models on 2D, 3D, and real-world datasets.

- **Hardware Simulations**  
  Simulate quantum circuits on a superconducting transmon processor with realistic noise profiles.

---

## Code Structure

### Quantum Neural Networks (`src/QNN`)

- **`BaseQNN`**  
  Abstract base class providing training, evaluation, and shared QNN utilities.

- **`GateQNN`**  
  Standard variational quantum model implemented with parameterized gates.

- **`PulsedQNN`**  
  Pulse-based variational model leveraging Trotterized time evolution.  
  Supports custom pulse encodings and hardware-inspired configurations.

### Pennypulse (`src/pennypulse`)

Pennylane is used to model quantum circuits. Some modifications were made to ensure compatibility with JAX operations used in our experiments. Key components:

- **`shapes/`** – Custom pulse shapes used in experiments.
- **`trotterization.py`** – Trotterized 2-qubit pulse simulations.
- **`pulses.py`** – Single-qubit pulse implementations.

### Experiments (`src/experiments`)

Scripts for running experiments on 2D, 3D, and real datasets. Experiments involve training and evaluating both **GateQNN** and **PulsedQNN**. Key scripts:

- **[`main.py`](main.py)** – Reproduces experiments varying noise levels.
- **[`layer_dependency.py`](layer_dependency.py)** – Studies performance with varying numbers of layers.

---

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/nacedob/Pulsed-Data-Reuploading-Quantum-Models.git
```

2. Install dependencies:

The recommended method is to use [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/):
```bash
cd Pulsed-Data-Reuploading-Quantum-Models
uv sync
```

Alternatively use the `requirements.txt` file:
```bash
cd Pulsed-Data-Reuploading-Quantum-Models
pip install -r requirements.txt
```

3. Run the main script:

If you have uv installed:
```bash
uv run main.py
```

Or use the standard `python` command:
```bash
python main.py
```
4. You should see the results in the terminal.

## Citation
If you use this code in your research, please cite:
```
@misc{acedo2025pulsedlearningquantumdata,
      title={Pulsed learning for quantum data re-uploading models}, 
      author={Ignacio B. Acedo and Pablo Rodriguez-Grasa and Pablo Garcia-Azorin and Javier Gonzalez-Conde},
      year={2025},
      eprint={2512.10670},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2512.10670}, 
}
```
## License

This code was entirely developed by **Ignacio Acedo**. You are free to use, modify, or fork it, provided that:

- Proper credit is given to the author.
- Any use is notified to the author

For any questions or clarifications, please feel free to contact me at the following email: [iacedo@ucm.es](mailto:iacedo@ucm.es).

## Contributing

Contributions are welcome! If you want to contribute to this project:

1. **Fork the repository** and create your branch:
   ```bash
   git checkout -b feature/your-feature
2. Make your changes following the existing code style.
3. Test your changes to ensure experiments still run correctly. Include them in the `tests` folder.
4. Submit a pull request with a clear description of your modifications.
# Pulsed-Data-Reuploading-Quantum-Models

This repository implements a framework for **Pulsed Quantum Machine Learning (PQML)** with **data re-uploading**, where traditional parameterized gates are replaced by hardware-efficient parameterized pulses. It includes pulse-based encoding techniques, benchmarking tools, and integration with pulse-aware quantum simulators.

## Features

- **Pulsed QML Framework**  
  Train quantum neural networks using parameterized pulse controls instead of gate sequences.

- **Pulse-Based Data Reuploading**  
  Encoding method that injects classical data into the model through pulse parameters, enabling richer representational capacity.


- **Comparisons with Gate-Based QML**  
  Scripts for benchmarking pulsed, gate-based, and hybrid models on 2D, 3D, and real datasets.

## Code Structure

### Quantum Neural Networks (`src/QNN`)

- **BaseQNN**  
  Abstract class defining training, evaluation, and common QNN utilities.

- **GateQNN**  
  Standard variational quantum model implemented with parameterized gates.

- **PulsedQNN**  
  Pulse-based variational model using Trotterized time evolution.  
  Supports custom pulse encodings and hardware-inspired configurations.


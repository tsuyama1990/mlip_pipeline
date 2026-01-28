# PyAcemaker: Automated MLIP Construction System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)

PyAcemaker is a "Zero-Config" workflow that automates the construction of Machine Learning Interatomic Potentials (MLIP) using the Atomic Cluster Expansion (ACE) framework.

## Overview

### What is PyAcemaker?
PyAcemaker democratizes the creation of state-of-the-art potentials. It bridges the gap between high-accuracy Density Functional Theory (DFT) and large-scale Molecular Dynamics (MD) by automating the tedious cycles of structure generation, calculation, training, and verification.

### Why use it?
- **Automated**: Handles the entire pipeline from active learning to potential fitting.
- **Data Efficient**: Uses active learning to select only the most informative structures, reducing DFT costs.
- **Robust**: Incorporates physical baselines (ZBL/LJ) and self-healing mechanisms.

## Features
- **Core Framework**: Robust configuration management and CLI.
- **DFT Oracle**: Automated interface to Quantum Espresso with error recovery.
- **Structure Generation**: Physics-informed generator (SQS, defects, distortions).
- **Dynamics Engine (LAMMPS)**:
  - Hybrid/Overlay potentials (ACE + ZBL/LJ) for safety.
  - On-the-fly uncertainty monitoring (`compute pace`).
  - Automatic halting on high uncertainty.
- **One-Shot Training**: Pipeline to generate, calculate, and train a potential in one go.
- **CLI Commands**: Easy-to-use commands for initialization, DFT execution, and validation.

## Requirements
- Python 3.11 or higher
- Quantum Espresso (`pw.x`) installed and in PATH (for DFT calculations)
- LAMMPS (`lmp` or similar) installed and in PATH (for MD simulations)
- `uv` package manager (recommended)

## Installation

```bash
git clone https://github.com/your-org/mlip-autopipec.git
cd mlip-autopipec
uv sync
```

## Usage

### 1. Initialize Project
Create a new project with a template configuration:
```bash
uv run mlip-auto init
```
This creates `input.yaml`. Edit it to specify your target system and DFT parameters.

### 2. Validate Configuration
Ensure your configuration is valid:
```bash
uv run mlip-auto validate input.yaml
```

### 3. Run DFT Calculation (Single Structure)
Run a DFT calculation on a specific structure file (e.g., `.cif`, `.xyz`):
```bash
uv run mlip-auto run-dft --config input.yaml --structure my_structure.cif
```

### 4. Run One-Shot Training
Execute the generation, calculation, and training pipeline:
```bash
uv run mlip-auto run cycle-02 --config input.yaml
```
Use `--mock-dft` to simulate DFT calculations for testing or if `pw.x` is unavailable.

### 5. Run Full Loop
```bash
uv run mlip-auto run loop --config input.yaml
```

## Architecture
```ascii
mlip_autopipec/
├── config/        # Pydantic schemas and loaders
├── dft/           # Quantum Espresso runner and error handling
├── inference/     # Dynamics Engine (LAMMPS/EON) and Uncertainty
├── app.py         # CLI entry point
└── ...
```

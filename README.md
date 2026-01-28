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
- **Physics-Aware**: Validates potentials against Phonon, Elastic, and EOS criteria.

## Features
- **Active Learning Loop**: Autonomous cycle of Exploration (MD) -> Detection (Uncertainty) -> Selection (D-Optimality) -> Calculation (DFT) -> Refinement (Training).
- **Core Framework**: Robust configuration management and CLI.
- **DFT Oracle**: Automated interface to Quantum Espresso with error recovery.
- **Structure Generation**: Physics-informed generator (SQS, defects, distortions).
- **Dynamics Engine (LAMMPS)**:
  - Hybrid/Overlay potentials (ACE + ZBL/LJ) for safety.
  - On-the-fly uncertainty monitoring (`compute pace`).
  - Automatic halting on high uncertainty.
- **Physics Validation**:
  - **Phonon**: Checks dynamical stability (imaginary frequencies).
  - **Elastic**: Calculates stiffness matrix and checks mechanical stability.
  - **EOS**: Fits Equation of State and checks thermodynamic behavior.
  - **Reporting**: Generates HTML reports with Pass/Fail metrics.
- **One-Shot Training**: Pipeline to generate, calculate, and train a potential in one go.

## Requirements
- Python 3.11 or higher
- Quantum Espresso (`pw.x`) installed and in PATH (for DFT calculations)
- LAMMPS (`lmp` or similar) installed and in PATH (for MD simulations)
- Pacemaker (`pacemaker`, `pace_activeset`) installed and in PATH (for training and selection)
- Phonopy (for phonon validation)
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
uv run mlip-auto validate --config input.yaml
```

### 3. Run Physics Validation
Validate a potential against physical criteria:
```bash
uv run mlip-auto validate --config input.yaml --potential my_pot.yace --phonon --elastic --eos
```

### 4. Run DFT Calculation (Single Structure)
Run a DFT calculation on a specific structure file (e.g., `.cif`, `.xyz`):
```bash
uv run mlip-auto run-dft --config input.yaml --structure my_structure.cif
```

### 5. Run Active Learning Loop
Execute the full autonomous active learning loop:
```bash
uv run mlip-auto run-loop --config input.yaml
```
This will run multiple cycles, exploring with MD and selecting uncertain structures for DFT refinement.

## Architecture
```ascii
mlip_autopipec/
├── config/        # Pydantic schemas and loaders
├── dft/           # Quantum Espresso runner and error handling
├── inference/     # Dynamics Engine (LAMMPS/EON) and Uncertainty
├── orchestration/ # Workflow management and active learning loop
├── surrogate/     # Candidate selection and active learning logic
├── validation/    # Physics validation suite (Phonon, Elastic, EOS)
├── app.py         # CLI entry point
└── ...
```

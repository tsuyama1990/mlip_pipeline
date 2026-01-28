# PyAcemaker (MLIP-AutoPipe)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

> **PyAcemaker**: A Python-based Automated ACE Maker for constructing Machine Learning Interatomic Potentials.

## Overview

**What:** PyAcemaker is an end-to-end pipeline for automating the generation of Machine Learning Interatomic Potentials (MLIPs), specifically Atomic Cluster Expansion (ACE) potentials.

**Why:** Creating robust MLIPs requires complex workflows involving structure generation, DFT calculations, active learning, and model training. PyAcemaker streamlines this process into a unified, reproducible, and scalable pipeline.

## Features

*   **Automated Workflow:** Orchestrates the entire lifecycle from structure generation to potential training.
*   **Active Learning:** Intelligent selection of training configurations to minimize DFT cost.
*   **DFT Automation:** Robust execution of Quantum Espresso calculations with error recovery.
*   **Physics Validation:** Integrated suite for validating potentials against Phonons, Elastic Constants, and EOS (Equation of State).
*   **Scalable Data Management:** SQLite-based storage with streaming support for large datasets.
*   **Surrogate Modeling:** Support for MACE and other surrogate models for pre-screening.
*   **Safety & Security:** Input validation and secure execution of external commands.

## Requirements

*   Python >= 3.12
*   Quantum Espresso (`pw.x`)
*   LAMMPS (`lmp`)
*   Pacemaker
*   EON (optional, for specific sampling)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mlip-autopipec.git
cd mlip-autopipec

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install .
```

## Usage

### 1. Initialize Project

Create a new project with a default configuration file:

```bash
mlip-runner init
```

Edit `input.yaml` to match your system requirements (elements, composition, DFT settings).

### 2. Run Validation

Validate an existing potential:

```bash
mlip-runner validate input.yaml --potential my_potential.yace --phonon --elastic --eos
```

### 3. Run Full Loop

Execute the continuous active learning loop:

```bash
mlip-runner run-loop input.yaml
```

### 4. Run One-Shot Cycle

Run a single generation cycle (Generation -> DFT -> Train):

```bash
mlip-runner run-cycle-02 input.yaml
```

## Architecture

```
src/mlip_autopipec/
├── config/           # Pydantic schemas and configuration loading
├── dft/              # DFT execution and parsing (Quantum Espresso)
├── domain_models/    # Core data models (Atoms, Results, State)
├── generator/        # Structure generation (Random, SQS, Defects)
├── inference/        # MD inference (LAMMPS, EON)
├── monitoring/       # Dashboard and metrics
├── orchestration/    # Workflow management and phases
├── surrogate/        # Surrogate models (MACE)
├── training/         # Potential training (Pacemaker)
├── utils/            # Utilities (Logging, ASE helpers)
└── validation/       # Physics validation suite
```

## Roadmap

*   Advanced Sampling Strategies (e.g., genetic algorithms).
*   Support for additional DFT codes (VASP, ABINIT).
*   Enhanced Web Dashboard for real-time monitoring.

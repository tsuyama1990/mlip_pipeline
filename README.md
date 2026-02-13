# PYACEMAKER

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Coverage](https://img.shields.io/badge/coverage-86%25-green)

**Zero-Configuration, Physics-Informed Active Learning for Machine Learning Interatomic Potentials.**

## Overview

**PYACEMAKER** automates the complex lifecycle of creating robust Machine Learning Interatomic Potentials (MLIPs). It bridges the gap between accurate Density Functional Theory (DFT) calculations and scalable Molecular Dynamics (MD) simulations.

### Why PYACEMAKER?
Constructing an MLIP typically requires expertise in multiple domains (DFT, MD, ML) and tedious manual workflow management. PYACEMAKER provides a "Zero-Config" solution that handles:
*   Intelligent structure generation (Active Learning).
*   Automated DFT calculations (Oracle).
*   Potential training and validation.
*   Self-healing workflow management.

## Features

*   **Physics Validation**: Automated validation suite ensuring potentials respect physical laws:
    *   **Equation of State (EOS)**: Fits Birch-Murnaghan curves to verify bulk modulus ($B_0$) and equilibrium volume ($V_0$).
    *   **Elastic Constants**: Calculates $C_{11}, C_{12}, C_{44}$ and shear/bulk moduli to ensure mechanical stability.
    *   **Phonon Stability**: Computes phonon dispersion (via Phonopy) to detect imaginary frequencies and dynamic instability.
    *   **HTML Reports**: Generates comprehensive validation reports with key metrics.
*   **Long-Timescale Evolution (aKMC)**: Integrates **EON (Adaptive Kinetic Monte Carlo)** to simulate rare events and diffusion processes over seconds or hours.
*   **Adaptive Structure Generation**: Automatically switches between Random, M3GNet (pre-trained), and MD-based exploration strategies.
*   **Automated DFT (Oracle)**:
    *   **Quantum Espresso Integration**: Generates input files and runs calculations via ASE.
    *   **Self-Healing**: Automatically recovers from SCF convergence failures.
    *   **Periodic Embedding**: Cuts local clusters from large MD snapshots for efficient DFT.
*   **Active Learning Loop**:
    *   **Dynamics Engine**: Runs MD using LAMMPS with hybrid potential overlay.
    *   **Uncertainty Watchdog**: Monitors extrapolation grade in real-time.
    *   **Halt & Diagnose**: Extracts problematic structures and triggers retraining autonomously.
*   **Pacemaker Integration**: Automates training of ACE potentials with Delta Learning.
*   **Zero-Config Automation**: Define your material system and let the system handle the rest.

## Requirements

*   Python >= 3.12
*   [Optional] Quantum Espresso (pw.x) in PATH for production Oracle
*   [Optional] LAMMPS (`lmp` executable) in PATH for production Dynamics
*   [Optional] Pacemaker (pace_train, pace_collect) for production Training
*   [Optional] EON (`eonclient` executable) in PATH for aKMC Simulations
*   [Optional] Phonopy for phonon validation

## Installation

```bash
git clone <repository_url>
cd mlip-pipeline
uv sync
```

## Usage

### 1. Initialize a Configuration

Create a default configuration file:

```bash
uv run mlip-runner init --output-path config.yaml
```

This will generate a YAML file with the new `validator` settings.

### 2. Run the Pipeline

Execute the workflow using the configuration file:

```bash
uv run mlip-runner run config.yaml
```

### 3. Check Outputs

Results (potentials, validation reports, logs) are saved in the directory specified in `config.yaml` (default: `experiments/`).

## Architecture

```ascii
src/mlip_autopipec/
├── core/               # Orchestrator, Active Learner, Config Parser, State Manager
├── domain_models/      # Pydantic Schemas (Config, Structure, Potential)
├── generator/          # Structure Generation (Adaptive, Candidate Generator)
├── oracle/             # DFT Engine Interfaces, Embedding, Self-Healing
├── trainer/            # Potential Training Logic
├── dynamics/           # MD (LAMMPS) & aKMC (EON) Drivers
├── validator/          # Physics Validation (EOS, Elastic, Phonon, Reports)
└── main.py             # CLI Entry Point
```

## Roadmap

*   **Cycle 08**: Production Validation (Completed).

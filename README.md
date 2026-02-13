# PYACEMAKER

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Coverage](https://img.shields.io/badge/coverage-88%25-green)

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

*   **Adaptive Structure Generation**: Automatically switches between Random, M3GNet (pre-trained), and MD-based exploration strategies based on the learning cycle.
*   **Automated DFT (Oracle)**:
    *   **Quantum Espresso Integration**: Generates input files and runs calculations via ASE.
    *   **Self-Healing**: Automatically recovers from SCF convergence failures by adjusting mixing beta and smearing.
    *   **Periodic Embedding**: Cuts local clusters from large MD snapshots and embeds them in periodic boxes for efficient DFT.
    *   **Parallel Execution**: Manages concurrent DFT tasks with robust error handling.
*   **Active Learning Loop (Cycle 05 Verified)**:
    *   **Dynamics Engine**: Runs Molecular Dynamics (MD) simulations using LAMMPS.
    *   **Hybrid Potential (Overlay)**: Enforces physical robustness by overlaying ACE potentials with ZBL or LJ baselines to prevent nuclear fusion.
    *   **Uncertainty Watchdog**: Monitors extrapolation grade ($\gamma$) in real-time and halts simulations automatically if the potential becomes unreliable.
*   **Pacemaker Integration**: Automates training of ACE potentials with Delta Learning and Active Set Selection.
*   **Zero-Config Automation**: Define your material system and let the system handle the rest.
*   **Robust Error Handling**: Centralized state management and self-healing capabilities.

## Requirements

*   Python >= 3.12
*   [Optional] Quantum Espresso (pw.x) in PATH for production Oracle
*   [Optional] LAMMPS (`lmp` executable) in PATH for production Dynamics
*   [Optional] Pacemaker (pace_train, pace_collect) for production Training

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

This will generate a YAML file with the new `oracle` and `dynamics` settings.

### 2. Run the Pipeline

Execute the workflow using the configuration file:

```bash
uv run mlip-runner run config.yaml
```

### 3. Check Outputs

Results (potentials, logs, state) are saved in the directory specified in `config.yaml` (default: `experiments/`).

## Architecture

```ascii
src/mlip_autopipec/
├── core/               # Orchestrator, Config Parser, State Manager, Logger
├── domain_models/      # Pydantic Schemas (Config, Structure, Potential)
├── generator/          # Structure Generation (Adaptive, M3GNet, Random)
├── oracle/             # DFT Engine Interfaces, Embedding, Self-Healing, Drivers
├── trainer/            # Potential Training Logic
├── dynamics/           # MD/MC Simulation Drivers (LAMMPS, Mock)
├── validator/          # Physics-based Validation Tests
└── main.py             # CLI Entry Point
```

## Roadmap

*   **Cycle 06**: Local Learning (Perturbation & Feedback).
*   **Cycle 07**: Long-timescale Evolution (aKMC with EON).
*   **Cycle 08**: Production Validation (Phonons, EOS).

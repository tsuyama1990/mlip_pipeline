# PYACEMAKER

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Coverage](https://img.shields.io/badge/coverage-91%25-green)

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
*   **Temperature Scheduling**: Implements simulated annealing protocols for efficient PES exploration.
*   **LAMMPS Integration**: Generates production-ready LAMMPS input scripts for Molecular Dynamics.
*   **Zero-Config Automation**: Define your material system and let the system handle the rest.
*   **Active Learning Loop**: Automatically explores chemical space and selects the most informative structures for labeling.
*   **Robust Error Handling**: Centralized state management and self-healing capabilities.

## Requirements

*   Python >= 3.12
*   [Optional] Quantum Espresso / VASP (for Production Oracle)
*   [Optional] LAMMPS (for Production Dynamics)

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

This will generate a YAML file with the new `generator.policy` settings.

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
├── oracle/             # DFT Engine Interfaces
├── trainer/            # Potential Training Logic
├── dynamics/           # MD/MC Simulation Drivers
├── validator/          # Physics-based Validation Tests
└── main.py             # CLI Entry Point
```

## Roadmap

*   **Cycle 03**: Quantum Espresso & VASP Integration.
*   **Cycle 04**: Pacemaker Training Integration.
*   **Cycle 05**: Uncertainty-driven Active Learning.
*   **Cycle 07**: Long-timescale Evolution (aKMC with EON).

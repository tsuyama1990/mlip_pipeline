# MLIP AutoPipeC

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)

> Automated construction of Machine Learning Interatomic Potentials (MLIP) using Active Learning.

## Overview

**What:** MLIP AutoPipeC is an orchestration framework that automates the generation of interatomic potentials. It integrates Structure Generation, First-Principles Calculations (DFT), and Machine Learning Potential Training (ACE) into a seamless, self-correcting loop.

**Why:** Creating robust MLIPs manually is error-prone and labor-intensive. This tool ensures D-Optimality (data efficiency) and physical stability (robustness) by automating the active learning cycle, allowing researchers to focus on materials science rather than workflow management.

## Features

*   **Long-Timescale Exploration (AKMC)**: Integrated EON interface for Adaptive Kinetic Monte Carlo to sample rare events and diffusion paths.
*   **Production Deployment**: Automatically packages the final potential with metadata manifests, validation reports, and licenses into a distributable ZIP.
*   **Active Learning with MD**: Integrated Molecular Dynamics engine (LAMMPS) for exploring phase space.
*   **On-the-Fly Safety Net**: Automatically detects high-uncertainty configurations ("Halt") during MD and AKMC runs, extracting them for labeling.
*   **Hybrid Potential**: Programmatically mixes Machine Learning potentials with physics-based baselines (ZBL) to ensure stability at short interatomic distances.
*   **Adaptive Exploration**: Automatically generates new candidate structures using smart policies (Strain, Defects) to explore the potential energy surface efficiently.
*   **Periodic Embedding**: Intelligent extraction of local defect environments into computable periodic supercells for DFT.
*   **Zero-Config Workflow**: Initialize complex pipelines with a single `config.yaml`.
*   **Robust State Management**: Automatic state persistence ensures jobs can be resumed after interruptions.
*   **Self-Healing DFT Oracle**: Built-in resilience for Quantum Espresso calculations, automatically retrying failed SCF cycles with adjusted parameters.
*   **Modular Architecture**: Plug-and-play components for Structure Generation, Oracle (DFT), and Training.
*   **Mock Mode**: Skeleton execution mode for rapid development and testing without expensive physics backends.
*   **Strict Validation**: Pydantic-based configuration ensures fail-fast behavior for invalid inputs.

## Requirements

*   Python 3.12+
*   `uv` (recommended) or `pip`
*   External Dependencies:
    *   `pw.x` (Quantum Espresso) - Required for DFT Oracle
    *   `pace_train` (Pacemaker) - Required for Training
    *   `lmp` (LAMMPS) - Required for MD Exploration
    *   `eonclient` (EON) - Required for AKMC Exploration

## Installation

```bash
git clone https://github.com/your-org/mlip-autopipec.git
cd mlip-autopipec
uv sync
```

## Tutorials

We provide Jupyter Notebook tutorials in the `tutorials/` directory to help you get started.

*   **01_quickstart_silicon.ipynb**: The "Zero-Config" experience running in Mock Mode.
*   **02_advanced_tio2.ipynb**: Demonstrates advanced configuration for complex oxide systems.
*   **03_validation_suite.ipynb**: A deep dive into the QA/Validation module.

To run these tutorials, ensure you have the dev dependencies installed:

```bash
uv sync --group dev
jupyter notebook tutorials/
```

## Usage

### 1. Minimal Mock Configuration (Quick Start)
To run the pipeline in **Mock Mode** (simulated physics, no external binaries required), set the environment variable `PYACEMAKER_MOCK_MODE=1`. This will force all components (Oracle, Explorer) to use their mock implementations regardless of the configuration file.

```bash
export PYACEMAKER_MOCK_MODE=1
uv run python -m mlip_autopipec.main config.yaml
```

Alternatively, you can configure mocks explicitly in `config.yaml`:

```yaml
project:
  name: "MockRun"
  seed: 42

training:
  dataset_path: "data.pckl"
  max_epochs: 5

orchestrator:
  max_iterations: 3

oracle:
  method: "mock"

exploration:
  strategy: "adaptive"
```

### 2. Production Configuration (DFT + ACE + Exploration + MD)
For production runs using Quantum Espresso, LAMMPS, and Adaptive Exploration:

```yaml
project:
  name: "TitaniumOxide"
  seed: 42

training:
  dataset_path: "/path/to/data.pckl"
  max_epochs: 100

orchestrator:
  max_iterations: 10

exploration:
  strategy: "adaptive"
  parameters:
    strain_range: 0.1
    defect_type: "vacancy"

oracle:
  method: "dft"

dft:
  command: "pw.x"
  pseudopotentials:
    Ti: "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF"
    O: "O.pbe-n-kjpaw_psl.1.0.0.UPF"
  ecutwfc: 50.0
  kspacing: 0.04

lammps:
  command: "lmp"
  num_processors: 4
```

### 3. Run the Pipeline

```bash
uv run python -m mlip_autopipec.main config.yaml
```

## Architecture

```
src/mlip_autopipec/
├── config/             # Pydantic Configuration Models
├── domain_models/      # Core Data Structures (WorkflowState, Potential, Structures, Production)
├── inference/          # Inference Drivers (EON Interface)
├── infrastructure/     # Deployment & Packaging
├── orchestration/      # State Machine & Main Loop
├── physics/            # Interfaces for Physics Engines
│   ├── dynamics/       # MD (LAMMPS) & AKMC (EON) Engines
│   ├── oracle/         # DFT Implementation (Quantum Espresso) with Self-Healing
│   ├── structure_gen/  # Exploration Logic (Generators, Embedding, Policy)
│   └── training/       # Potential Training (Pacemaker)
└── utils/              # Shared Utilities (Parsers, File Ops, Logging)
```

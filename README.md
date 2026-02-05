# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an automated system for constructing robust Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) formalism. It provides a structured workflow to iterate through structure generation, calculation, training, and validation.

## Overview

-   **What**: A modular Python pipeline for automating the training of interatomic potentials.
-   **Why**: Manual training of MLIPs is tedious and error-prone. This tool automates the active learning loop, ensuring robustness and reproducibility.

## Features

-   **Modular Architecture**: Interface-driven design allowing easy swapping of components (Explorer, Oracle, Trainer).
-   **Strict Configuration**: Validated `config.yaml` using Pydantic ensures inputs are correct before execution.
-   **Scalable Design**: Optimized for batch processing and large datasets using lazy loading abstractions.
-   **Mock Mode**: Fully functional simulation of the pipeline using Mock components for rapid development and testing without heavy physics engines.
-   **Automated Loop**: Orchestrates the cycle of Exploration -> Calculation -> Training -> Validation.
-   **Comprehensive Logging**: Structured logging for full traceability of the pipeline execution.

## Requirements

-   **Python 3.12+**
-   **uv** (Modern Python package manager)

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd mlip-pipeline

# Install dependencies
uv sync
```

## Usage

### 1. Create a Configuration File

Create a file named `config.yaml`:

```yaml
work_dir: "./workspace"
logging_level: "INFO"
max_cycles: 2
random_seed: 42

exploration:
  strategy: "random"
  max_structures: 10

dft:
  calculator: "mock"

training:
  trainer: "mock"
  epochs: 5
```

### 2. Run the Pipeline

```bash
uv run mlip-pipeline run config.yaml
```

The pipeline will execute the defined number of cycles using Mock components (in the current version), generating a `potential.yace` file and a validation report in the `workspace` directory.

## Architecture/Structure

```ascii
src/mlip_autopipec/
├── config/                  # Pydantic Configuration Models
├── domain_models/           # Strict Domain Entities (Structures, Dataset)
├── interfaces/              # Protocol Definitions (Explorer, Oracle, Trainer, etc.)
├── orchestration/           # Control Logic (Orchestrator, Mocks)
├── services/                # Concrete Implementations (Future)
└── utils/                   # Shared Utilities (Logging)
```

## Roadmap

-   **Cycle 02**: Integration with real Pacemaker training and physical baselines.
-   **Cycle 03**: Connection to Quantum Espresso (Oracle) with self-healing.
-   **Cycle 04**: Advanced structure generation (MD/MC/Defects).
-   **Cycle 05**: Dynamics Engine integration (LAMMPS) and Uncertainty quantification.
-   **Cycle 06**: Full system validation (Phonons, Elasticity) and Scale-up.

# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous system for constructing and operating State-of-the-Art Machine Learning Interatomic Potentials (MLIP) using the Pacemaker (ACE) framework.

## Overview
**What**: A Python-based orchestration framework for Active Learning loops in materials science.
**Why**: To automate the complex workflow of structure generation, DFT calculation, and potential training.

## Features
*   **Mock Active Learning Loop**: A complete skeleton of the active learning cycle (Explore -> Label -> Train -> Validate) using mock components for verification.
*   **Schema-First Design**: Robust data validation using Pydantic for all domain models and configurations.
*   **Secure Configuration**: Strict validation of configuration files and file paths.
*   **Encapsulated Data Handling**: Abstractions for Dataset management to support future scalability.
*   **CLI Interface**: Easy-to-use command line interface powered by Typer.

## Requirements
*   Python 3.12+
*   `uv` (recommended) or `pip`

## Installation

```bash
git clone https://github.com/your-org/mlip-pipeline.git
cd mlip-pipeline
uv sync
```

## Usage

To run the pipeline with the provided mock configuration:

1.  **Create a config file** (e.g., `config.yaml`):
    ```yaml
    work_dir: "/tmp/mlip_work"
    max_cycles: 5
    random_seed: 42
    ```

2.  **Run the pipeline**:
    ```bash
    uv run mlip-pipeline run --config config.yaml
    ```

## Architecture

```ascii
src/mlip_autopipec/
├── config/             # Configuration schemas (GlobalConfig)
├── domain_models/      # Data structures (StructureMetadata, Dataset, ValidationResult)
├── interfaces/         # Core Protocols (Explorer, Oracle, Trainer, Validator)
├── orchestration/      # Main Loop & Mocks
├── utils/              # Utilities (Logging)
└── main.py             # CLI Entry Point
```

## Roadmap
*   Cycle 02: Trainer implementation (Pacemaker)
*   Cycle 03: Oracle implementation (Quantum Espresso)
*   Cycle 04: Dynamics engine integration
*   Cycle 05: Adaptive Structure Generation
*   Cycle 06: Validation & Scale-up

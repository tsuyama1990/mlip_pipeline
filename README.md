# PYACEMAKER (mlip-pipeline)

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**The Zero-Configuration Autonomous Pipeline for Machine Learning Interatomic Potentials.**

## Overview

### What is this?
PYACEMAKER is an intelligent orchestrator that automates the construction of Machine Learning Interatomic Potentials (specifically ACE - Atomic Cluster Expansion). It closes the loop between Structure Generation (Exploration), First-Principles Calculations (DFT), and Model Training (Pacemaker).

### Why use it?
Building ML potentials manually is tedious and error-prone. PYACEMAKER provides a "Fire and Forget" solution: define your material in a simple config file, and the system autonomously explores the chemical space, runs DFT, trains the model, and validates it.

## Features

-   **Autonomous Orchestration**: Manages the cyclic workflow of Explore -> Label -> Train -> Validate.
-   **Strict Configuration**: Uses strict schema validation (Pydantic) to prevent misconfiguration.
-   **Modular Architecture**: Defines clear protocols for `Explorer`, `Oracle`, `Trainer`, and `Validator`, allowing easy component swapping.
-   **Logging**: Centralized, structured logging for full traceability.
-   **Mock Mode**: currently runs in a "Walking Skeleton" mode using Mock components for verification of the pipeline logic (Cycle 01).

## Requirements

-   **Python**: 3.12+
-   **Package Manager**: `uv` (Recommended)
-   **Dependencies**: `ase`, `typer`, `pydantic`, `pyyaml`.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install Dependencies**
    Using `uv` is recommended for fast, deterministic installation:
    ```bash
    uv sync
    ```

3.  **Install in Editable Mode**
    ```bash
    uv pip install -e .
    ```

## Usage

### Basic Execution
To run the pipeline using the provided example configuration:

```bash
uv run python -m mlip_autopipec.main run config.yaml
```

You should see output indicating the progress of the mock cycle:
```text
[INFO] Loading configuration from config.yaml
[INFO] PYACEMAKER initialized for project: Cycle01-Demo
[INFO] Starting orchestration cycle...
...
[INFO] Cycle completed successfully.
```

### Configuration
Edit `config.yaml` to customize the run. Example:
```yaml
project_name: "MyMaterial"
dft:
  code: "qe"
  ecutwfc: 40.0
  kpoints: [2, 2, 2]
training:
  code: "pacemaker"
  cutoff: 5.0
```

### Help
```bash
uv run python -m mlip_autopipec.main --help
uv run python -m mlip_autopipec.main run --help
```

## Architecture/Structure

```text
src/mlip_autopipec/
├── config/             # Pydantic configuration models
├── domain_models/      # Data schemas (ValidationResult, etc.)
├── interfaces/         # Core Protocols (Explorer, Oracle, etc.)
├── orchestration/      # Workflow logic (Orchestrator) & Mocks
├── physics/            # Scientific implementations (Placeholder)
├── utils/              # Logging and utilities
└── main.py             # CLI entry point
```

## Roadmap

-   **Cycle 01**: Core Framework & Infrastructure (Completed)
-   **Cycle 02**: The Oracle (DFT Automation)
-   **Cycle 03**: The Explorer (Structure Generation)
-   **Cycle 04**: The Trainer (Pacemaker Integration)
-   **Cycle 05**: Active Learning Loop (OTF Dynamics)
-   **Cycle 06**: Production Readiness & Validation

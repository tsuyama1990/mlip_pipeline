# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous system for constructing and operating State-of-the-Art Machine Learning Interatomic Potentials (MLIP). It automates the "Generation -> Calculation -> Training" loop.

## Overview

### What is this?
A modular Python framework to orchestrate Active Learning cycles for materials science.

### Why?
To reduce the manual effort in training ML potentials by automating data generation, labeling (DFT), and training.

## Features (Verified Cycle 01)

-   **Core Orchestration**: A robust loop managing the active learning cycle.
-   **Configurable Workflow**: YAML-based configuration for experiment parameters.
-   **Mock Loop**: A verified skeleton verifying the data flow between Explorer, Oracle, and Trainer.
-   **Structured Logging**: Comprehensive logging for process monitoring.
-   **Type-Safe Architecture**: Strict Pydantic models for all data structures (Structures, Datasets, Configs).

## Requirements

-   **Python**: 3.12 or higher.
-   **Package Manager**: `uv` (recommended).

## Installation

```bash
git clone https://github.com/your-org/mlip-pipeline.git
cd mlip-pipeline
uv sync
```

## Usage

### Running the Mock Loop
Create a `config.yaml` file:

```yaml
work_dir: "./workspace"
max_cycles: 3
random_seed: 42
```

Run the pipeline:

```bash
export PYTHONPATH=src
uv run python src/main.py run --config config.yaml
```

You should see logs indicating the progression of cycles (Generation -> Calculation -> Training).

## Architecture / Structure

```ascii
src/
├── main.py                     # CLI Entry Point
├── config/                     # Configuration Schemas
├── domain_models/              # Data Models (StructureMetadata, Dataset)
├── interfaces/                 # Core Protocols (Explorer, Oracle, Trainer)
├── orchestration/              # Orchestrator & Mock Components
└── utils/                      # Utilities (Logging)
```

## Roadmap

-   **Cycle 02**: Integrate Real Trainer (Pacemaker).
-   **Cycle 03**: Integrate Real Oracle (Quantum Espresso).
-   **Cycle 04**: Integrate Dynamics (LAMMPS).

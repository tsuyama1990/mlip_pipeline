# MLIP Pipeline (PYACEMAKER)

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

**Automated Machine Learning Interatomic Potential Pipeline**

## Overview

### What
MLIP Pipeline is an automated workflow system for generating and validating Machine Learning Interatomic Potentials (MLIPs). It orchestrates the Active Learning cycle: exploring structures (Molecular Dynamics), labeling them (DFT/Oracle), and training potentials.

### Why
Developing MLIPs requires complex iterative loops of data generation and training. This tool automates the process, ensuring reproducibility and scalability.

## Features

*   **Domain Models**: Strict Pydantic schemas for `Structure`, `Potential`, and `Config`.
*   **Modular Architecture**: Abstract Interfaces for `Oracle`, `Trainer`, `Dynamics`, `Generator`, `Validator`, and `Selector`.
*   **Mock Implementations**: Fully functional mock components for testing pipeline logic without expensive physics codes.
*   **CLI**: User-friendly command-line interface for initialization and execution.
*   **Configuration**: YAML-based configuration with strict validation.

## Requirements

*   Python >= 3.12
*   `uv` (Universal Package Manager) or `pip`

## Installation

```bash
git clone https://github.com/your-org/mlip-pipeline.git
cd mlip-pipeline
uv sync
uv pip install -e .
```

## Usage

### 1. Initialize Configuration
Generate a default `config.yaml`:
```bash
uv run mlip-pipeline init --path config.yaml
```

### 2. Run Pipeline
Execute the pipeline (currently runs mock loop):
```bash
uv run mlip-pipeline run --config config.yaml
```

### 3. Help
```bash
uv run mlip-pipeline --help
```

## Architecture

```
src/mlip_autopipec/
├── domain_models/    # Data schemas (Structure, Config, Potential)
├── interfaces/       # Abstract Base Classes (Oracle, Trainer, etc.)
├── infrastructure/   # Concrete Implementations (Mocks)
├── utils/            # Utilities (Logging)
└── main.py           # CLI Entry Point
```

## Roadmap

*   [x] Cycle 01: Foundation & Mocks
*   [ ] Cycle 02: Orchestration Logic
*   [ ] Cycle 03: Quantum Espresso Integration
*   [ ] Cycle 04: MACE/Allegro Training Integration
*   [ ] Cycle 05: LAMMPS/Rattle Dynamics
*   [ ] Cycle 06: Production Readiness

# MLIP Pipeline

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**The Zero-Configuration Autonomous Pipeline for Machine Learning Interatomic Potentials.**

## Overview

### What
MLIP Pipeline is a robust orchestration framework designed to automate the development of Machine Learning Interatomic Potentials (MLIPs). It integrates structure generation, First-Principles calculations (DFT), and potential training into a unified active learning loop.

### Why
Developing MLIPs typically involves managing complex, manual workflows across multiple software packages (LAMMPS, Quantum Espresso, Pacemaker). This tool creates a "Self-Driving" pipeline that handles data generation, training, and validation autonomously, reducing human error and accelerating discovery.

## Features

-   **Robust Configuration**: Strict schema validation ensures your experiment settings are always correct before execution.
-   **Modular Architecture**: Built on clean interfaces, allowing easy swapping of DFT codes (Quantum Espresso, VASP) or ML trainers.
-   **Orchestration**: Automated workflow management connecting exploration, labeling, training, and validation phases.
-   **Developer Experience**: Fully typed codebase with comprehensive logging and error handling.
-   **Mock Mode**: Built-in mock components allow verifying the workflow logic without expensive backend codes installed.

## Requirements

-   **Python**: 3.12+
-   **Package Manager**: `uv` (Recommended) or `pip`
-   **Operating System**: Linux or macOS

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mlip-pipeline.git
cd mlip-pipeline

# Install dependencies using uv
uv sync
```

## Usage

1.  **Prepare Configuration**
    Copy the example configuration file:
    ```bash
    cp config.example.yaml config.yaml
    ```

2.  **Run the Pipeline**
    Execute the workflow using the CLI:
    ```bash
    uv run mlip-pipeline run config.yaml
    ```

    You should see output indicating the successful execution of the orchestration cycle (Mock mode by default).

## Architecture/Structure

```text
src/mlip_autopipec/
├── config/             # Configuration schemas (Pydantic)
├── domain_models/      # Data transfer objects
├── interfaces/         # Core abstract protocols
├── orchestration/      # Workflow logic and State Machine
├── physics/            # Scientific implementations (DFT, MD)
├── utils/              # Logging and helpers
└── validation/         # Verification logic
```

## Roadmap

-   **Cycle 02**: Integration of Real DFT (Quantum Espresso) Oracle.
-   **Cycle 03**: Advanced Structure Exploration Strategies.
-   **Cycle 04**: Pacemaker Training Integration.
-   **Cycle 05**: Full Active Learning Loop with LAMMPS.
-   **Cycle 06**: Deployment and Production Readiness.

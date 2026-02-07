# PYACEMAKER: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous "Robot Scientist" for materials physics. It automates the construction of State-of-the-Art Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) formalism. By integrating Active Learning, Density Functional Theory (DFT) automation, and Physics-Informed constraints, it enables researchers to generate robust potentials for complex alloys with **Zero Configuration**.

> **Elevator Pitch:** "Input a chemical composition, get a production-ready, validated interatomic potential in days, not months—without writing a single line of code."

## Overview

PYACEMAKER aims to democratize the creation of MLIPs. Traditionally, fitting a potential requires expert knowledge of DFT, basis sets, and fitting algorithms. This tool replaces the "Human-in-the-Loop" with a "Physics-in-the-Loop" architecture, autonomously exploring chemical space and refining the potential until it meets strict quality standards.

## Current Features (Cycle 01)

The system currently implements the **Foundation & Orchestrator Skeleton**:

*   **Orchestration Logic**: A robust "Check-Decide-Act" loop that coordinates Exploration, Labeling, and Training.
*   **Mock Backend**: Fully functional simulation mode using Mock components (Oracle, Trainer, Explorer) to verify control flow without expensive physics calculations.
*   **Configuration System**: Type-safe configuration management using Pydantic, ensuring valid inputs before execution.
*   **Extensible Architecture**: Defined Abstract Base Classes (Interfaces) for easy integration of future engines (LAMMPS, Quantum Espresso, Pacemaker).
*   **Logging & Diagnostics**: Centralized logging system for tracking the autonomous process.

## Requirements

*   **Python 3.12+**
*   **uv** (Fast Python package manager)

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Initialize Environment**
    ```bash
    uv sync
    ```

3.  **Install in Editable Mode**
    ```bash
    uv pip install -e .
    ```

## Usage

To run the system in "Mock Mode" (verifying the loop):

1.  **Create a Configuration File**
    Create a file named `config.yaml` with the following content:
    ```yaml
    work_dir: "./simulation_run"
    max_cycles: 3
    oracle:
      type: "mock"
    trainer:
      type: "mock"
    explorer:
      type: "mock"
    ```

2.  **Run the Orchestrator**
    ```bash
    uv run python -m mlip_autopipec.main config.yaml
    ```

    You should see output indicating the start of cycles, mock exploration, extraction of candidates, mock training, and validation.

## Architecture

The project follows a modular Hexagonal Architecture:

```ascii
src/
└── mlip_autopipec/
    ├── config/             # Pydantic configuration models
    ├── domain_models/      # Core data structures (Dataset, Potential)
    ├── infrastructure/     # Implementations (Mocks, Adapters)
    ├── interfaces/         # Abstract Base Classes (Oracle, Trainer, Explorer)
    ├── main.py             # CLI and Orchestrator Logic
    └── utils/              # Logging and helper utilities
```

## Roadmap

*   **Cycle 02**: Espresso Oracle (Real DFT with Quantum Espresso).
*   **Cycle 03**: Structure Generator (Adaptive Exploration Policies).
*   **Cycle 04**: Pacemaker Trainer (Real MLIP fitting).
*   **Cycle 05**: Dynamics Engine (LAMMPS Integration).
*   **Cycle 06**: Active Learning Loop (Uncertainty-driven Halting).
*   **Cycle 07**: Kinetic Monte Carlo (EON Integration).
*   **Cycle 08**: Full Validation & Production Readiness.

## License

MIT License. See `LICENSE` for details.

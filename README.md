# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: The Foundation

This first cycle establishes the architectural backbone of the project. It includes:

- **Pydantic Schemas**: A robust, schema-driven framework for managing all system configurations.
- **DFT Factory Core**: The initial implementation of the `QEProcessRunner`, a component for interacting with the Quantum Espresso DFT engine.
- **Database Manager**: A data persistence layer for handling ASE-compatible databases with custom metadata.

## Cycle 02: The Seed - Physics-Informed Generator

Cycle 02 introduces the `PhysicsInformedGenerator`, a module that solves the "cold-start" problem in MLIP creation. It generates a diverse initial dataset of atomic structures without relying on prior DFT calculations.

- **Alloy Generation**: Creates Special Quasirandom Structures (SQS) and applies volumetric/shear strains and atomic rattling to explore the potential energy surface.
- **Crystal Defect Generation**: Systematically introduces point defects (vacancies, interstitials) into pristine crystal structures.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

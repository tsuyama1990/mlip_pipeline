# AutoMLIP: Automated Machine Learning Interatomic Potential Pipeline

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

> Automated generation, training, and validation of Machine Learning Interatomic Potentials (MLIPs).

## Overview

**AutoMLIP** is an autonomous pipeline designed to accelerate the development of interatomic potentials. It orchestrates the cycle of structure generation, active learning simulations, and model training.

**Why?** Developing MLIPs manually is tedious and error-prone. This tool automates the "loop" of exploring configuration space (via MD), labeling structures (via DFT/Oracle), and retraining models.

## Features

-   **Cycle 02: One-Shot Pipeline**: Execute a complete MD workflow from a configuration file.
    -   Generate bulk structures (e.g., Silicon Diamond).
    -   Apply thermal noise (rattle) to initial configurations.
    -   Run Molecular Dynamics simulations using LAMMPS.
    -   Parse and report simulation results (trajectories and final structures).
-   **Configurable**: YAML-based configuration for Potentials, MD parameters, and Resources.
-   **Robust Execution**: Handles timeouts, errors, and creates persistent job directories.

## Requirements

-   **Python 3.11+**
-   **LAMMPS**: The `lmp` (or similar) executable must be in your PATH or configured in `config.yaml`.
-   **UV**: Recommended for dependency management.

## Installation

```bash
git clone https://github.com/your-org/auto-mlip.git
cd auto-mlip
uv sync
```

## Usage

### 1. Initialize a Project
Create a default configuration file.

```bash
uv run mlip-auto init
```

### 2. Configure
Edit `config.yaml` to set your desired potential and LAMMPS settings.

```yaml
project_name: "Si_Exploration"
potential:
  elements: ["Si"]
  cutoff: 5.0
lammps:
  command: "lmp_serial"  # Path to your LAMMPS executable
  cores: 4
```

### 3. Run the One-Shot Pipeline
Execute the Cycle 02 workflow.

```bash
uv run mlip-auto run-cycle-02
```

The system will:
1.  Build a Supercell of Silicon.
2.  Rattle the atoms.
3.  Run an NVT MD simulation.
4.  Output the final structure and status.

## Architecture

```
src/mlip_autopipec/
├── domain_models/      # Pydantic schemas (Config, Job, Structure)
├── physics/            # Core physics logic
│   ├── dynamics/       # MD wrappers (LAMMPS)
│   └── structure_gen/  # Structure builders (ASE wrapper)
├── orchestration/      # Workflow managers
└── infrastructure/     # I/O, Logging
```

# PyAceMaker: Automated MLIP Pipeline

![Status](https://img.shields.io/badge/Status-Cycle_02_Exploration-blue)
![Python](https://img.shields.io/badge/Python-3.12%2B-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**PyAceMaker** is an autonomous research system designed to construct State-of-the-Art Machine Learning Interatomic Potentials (MLIPs). It democratises computational materials science by providing a "Zero-Config" workflow.

## Overview
**What**: A robust, type-safe pipeline for automating the generation of MLIPs.
**Why**: Scientific software often suffers from loose data contracts. This project establishes a rigid **Schema-First** foundation to prevent "silent failures" in complex simulations.

## Key Features

-   **One-Shot Pipeline (Cycle 02)**:
    -   **Automated MD**: Drive LAMMPS simulations directly from Python.
    -   **Structure Generation**: Create bulk supercells and apply thermal noise (rattling).
    -   **Trajectory Parsing**: Automatically convert LAMMPS outputs back to structured data.
-   **Strict Data Validation**: Pydantic-based domain models ensure that every atomic structure and configuration parameter is valid before processing begins.
-   **Configuration Management**:
    -   `init`: Generates valid template configurations.
    -   `check`: Validates existing configurations against strict schemas.
-   **Robust Infrastructure**:
    -   Type-safe YAML input/output.
    -   Dual-channel logging: Console (Rich) and File.

## Requirements

-   **Python**: 3.12+
-   **Package Manager**: `uv` (recommended)
-   **Physics Engine**: [LAMMPS](https://www.lammps.org/) (executable `lmp_serial` or `lmp_mpi` in PATH)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/mlip-autopipec.git
cd mlip-autopipec

# 2. Install dependencies
uv sync

# 3. (Optional) Activate virtual environment
source .venv/bin/activate
```

## Usage

### 1. Initialize a Project
Create a new project with a template configuration file.
```bash
uv run mlip-auto init
# Creates 'config.yaml' in the current directory
```

### 2. Run the One-Shot Pipeline (Cycle 02)
Execute the structure generation and MD simulation workflow.
```bash
uv run mlip-auto run-cycle-02 --config config.yaml
```
Expected output:
```text
Generating structure...
Running LAMMPS in _work_md/job_oneshot...
Simulation Completed: Status DONE
Final Energy: -123.45 eV
```

### 3. Validate Configuration
Check if your configuration file is valid.
```bash
uv run mlip-auto check --config config.yaml
```

## Project Structure

```ascii
src/mlip_autopipec/
├── cli/                    # Command Logic
├── domain_models/          # Pydantic Schemas (Structure, Config)
├── infrastructure/         # Logging, IO
├── modules/                # Core Logic (Structure Gen, Dynamics)
└── app.py                  # CLI Entry Point
```

## Roadmap

-   **Cycle 01**: Foundation & Core Models (Completed)
-   **Cycle 02**: Basic Exploration (MD) (Completed)
-   **Cycle 03**: Oracle (DFT)
-   **Cycle 04**: Training (Pacemaker)
-   **Cycle 05**: Validation Framework
-   **Cycle 06**: Active Learning Loop

## License

MIT License.

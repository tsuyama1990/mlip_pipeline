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
    -   **Automated MD Execution**: Runs Molecular Dynamics simulations via LAMMPS directly from Python.
    -   **Structure Generation**: Creates bulk crystal structures with thermal noise (rattle) and strain.
    -   **Robust Parsers**: Automatically parses LAMMPS log files and trajectory dumps into strict Python objects.
-   **Strict Data Validation**: Pydantic-based domain models ensure that every atomic structure and configuration parameter is valid before processing begins.
-   **Configuration Management**:
    -   `init`: Generates valid template configurations instantly.
    -   `check`: Validates existing configurations against strict schemas (e.g., catching negative cutoff radii).
-   **Robust Infrastructure**:
    -   Type-safe YAML input/output.
    -   Dual-channel logging: Beautiful console output (Rich) for users, detailed file logs for debugging.

## Requirements

-   **Python**: 3.12+
-   **Package Manager**: `uv` (recommended)
-   **LAMMPS**: `lmp_serial` or `lmp_mpi` executable (for MD simulations)

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

### 2. Validate Configuration
Check if your configuration file is valid.
```bash
uv run mlip-auto check --config config.yaml
```

### 3. Run One-Shot Simulation
Execute a single Molecular Dynamics run (Generate -> Run -> Parse).
```bash
uv run mlip-auto run-cycle-02 --config config.yaml
```

Example `config.yaml`:
```yaml
project_name: "MyMLIPProject"
potential:
  elements: ["Si"]
  cutoff: 5.0
exploration:
  lattice_constant: 5.43
  md_params:
    temperature: 300.0
    timestep: 0.001
    n_steps: 100
lammps:
  command: "lmp_serial"
  cores: 1
```

## Project Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Schemas (Structure, Config, Job)
├── physics/                # Physics Logic (Dynamics, Structure Gen)
├── orchestration/          # Workflow Management
├── infrastructure/         # Logging, IO
└── cli/                    # CLI Commands
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

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
    -   Automated structure generation (Bulk crystals + Rattle).
    -   Seamless integration with LAMMPS for Molecular Dynamics.
    -   Robust input generation and output parsing for MD trajectories.
-   **Strict Data Validation**: Pydantic-based domain models ensure that every atomic structure and configuration parameter is valid before processing begins.
-   **Configuration Management**:
    -   `init`: Generates valid template configurations instantly.
    -   `check`: Validates existing configurations against strict schemas.
-   **Robust Infrastructure**:
    -   Type-safe YAML input/output.
    -   Dual-channel logging: Beautiful console output (Rich) for users, detailed file logs for debugging.

## Requirements

-   **Python**: 3.12+
-   **Package Manager**: `uv` (recommended)
-   **External Tools**: `LAMMPS` (executable `lmp_serial` or configurable)

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

### 2. Run One-Shot Pipeline
Execute a single cycle of structure generation and MD simulation.
```bash
uv run mlip-auto run-cycle-02 --config config.yaml
```

Example `config.yaml` for Cycle 02:
```yaml
project_name: "Si_MD"
potential:
  elements: ["Si"]
  cutoff: 5.0
structure_gen:
  element: "Si"
  crystal_structure: "diamond"
  lattice_constant: 5.43
  supercell: [3, 3, 3]
lammps:
  command: "lmp_serial"
  timeout: 3600.0
```

## Project Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Schemas (Structure, Config, Job)
├── infrastructure/         # Logging, IO (LAMMPS Helpers)
├── modules/
│   └── structure_gen/      # Generators and Strategies
├── cli/                    # Command Handlers
└── app.py                  # CLI Entry Point
```

## Roadmap

-   **Cycle 01**: Foundation & Core Models (Completed)
-   **Cycle 02**: Basic Exploration (One-Shot Pipeline) (Completed)
-   **Cycle 03**: Oracle (DFT)
-   **Cycle 04**: Training (Pacemaker)
-   **Cycle 05**: Validation Framework
-   **Cycle 06**: Active Learning Loop

## License

MIT License.

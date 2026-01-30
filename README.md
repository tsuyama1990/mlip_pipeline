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
    -   Automated generation of atomic structures (Bulk + Rattle).
    -   Seamless interface with LAMMPS for Molecular Dynamics.
    -   Robust output parsing and error handling.
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
-   **External Tools**: `LAMMPS` (for MD simulations)

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
Create a new project with a template configuration file (including LAMMPS settings).
```bash
uv run mlip-auto init
# Creates 'config.yaml' in the current directory
```

### 2. Validate Configuration
Check if your configuration file is valid.
```bash
uv run mlip-auto check --config config.yaml
```

### 3. Run One-Shot Pipeline (Cycle 02)
Execute a single MD simulation pipeline.
```bash
uv run mlip-auto run-cycle-02
```
This command will:
1. Generate a structure based on `structure_gen` config.
2. Create a working directory `_work_md/job_001`.
3. Run LAMMPS using settings in `lammps` config.
4. Parse the results and report status.

Example `config.yaml`:
```yaml
project_name: "MyMLIPProject"
potential:
  elements: ["Ti", "O"]
  cutoff: 5.0
  seed: 42
lammps:
  command: "lmp_serial"
  cores: 1
  timeout: 3600.0
structure_gen:
  element: "Ti"
  crystal_structure: "hcp"
  lattice_constant: 2.95
  supercell: [2, 2, 2]
  rattle_stdev: 0.01
logging:
  level: "INFO"
  file_path: "mlip_pipeline.log"
```

## Project Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Schemas (Structure, Config, Job)
├── modules/                # Core Logic
│   └── structure_gen/      # Structure Generation & Strategies
├── infrastructure/         # Logging, IO
├── cli/                    # CLI Commands
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

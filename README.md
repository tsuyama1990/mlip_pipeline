# PyAceMaker: Automated MLIP Pipeline

![Status](https://img.shields.io/badge/Status-Cycle_03_Verified-green)
![Python](https://img.shields.io/badge/Python-3.12%2B-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**PyAceMaker** is an autonomous research system designed to construct State-of-the-Art Machine Learning Interatomic Potentials (MLIPs). It democratises computational materials science by providing a "Zero-Config" workflow.

## Overview
**What**: A robust, type-safe pipeline for automating the generation of MLIPs.
**Why**: Scientific software often suffers from loose data contracts. This project establishes a rigid **Schema-First** foundation to prevent "silent failures" in complex simulations.

## Key Features

-   **Oracle (DFT Automation)**:
    -   **Quantum Espresso** wrapper with self-healing capabilities.
    -   Automatic recovery from SCF convergence failures (adjusting mixing beta, smearing).
    -   Automatic K-point grid generation based on `kspacing`.
    -   **Periodic Embedding**: Extraction of representative clusters from large MD structures.
-   **Molecular Dynamics Engine**:
    -   Automated "One-Shot" MD pipelines via LAMMPS.
    -   Robust wrapper with input generation, execution management, and trajectory parsing.
    -   Graceful handling of missing executables and timeouts.
-   **Structure Generation**:
    -   Deterministic bulk crystal generation (e.g., Silicon Diamond).
    -   Thermal noise application (Rattling).
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
-   **Simulation Engine**: LAMMPS (Optional for core functionality, required for MD)
    -   Executable `lmp_serial` or `mpirun` accessible in PATH.

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
# Output: Configuration valid (or detailed error messages)
```

### 3. Run One-Shot MD
Execute a single Molecular Dynamics simulation (Generate -> MD -> Parse).
```bash
uv run mlip-auto run-one-shot --config config.yaml
```

Example `config.yaml`:
```yaml
project_name: "MyMLIPProject"
potential:
  elements: ["Si"]
  cutoff: 5.0
  seed: 42
lammps:
  command: "lmp_serial"
  timeout: 3600
  use_mpi: false
logging:
  level: "INFO"
  file_path: "mlip_pipeline.log"
```

## Project Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Schemas (Structure, Config, Job)
├── physics/                # Physics Engines (LAMMPS, StructureGen)
├── orchestration/          # Workflow Management
├── infrastructure/         # Logging, IO
└── app.py                  # CLI Entry Point
```

## Roadmap

-   **Cycle 01**: Foundation & Core Models (Completed)
-   **Cycle 02**: Basic Exploration (MD) (Completed)
-   **Cycle 03**: Oracle (DFT) (Completed)
-   **Cycle 04**: Training (Pacemaker)
-   **Cycle 05**: Validation Framework
-   **Cycle 06**: Active Learning Loop

## License

MIT License.

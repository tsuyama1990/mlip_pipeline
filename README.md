# PyAceMaker: Automated MLIP Pipeline

![Status](https://img.shields.io/badge/Status-Cycle_02_Exploration-blue)
![Python](https://img.shields.io/badge/Python-3.12%2B-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**PyAceMaker** is an autonomous research system designed to construct State-of-the-Art Machine Learning Interatomic Potentials (MLIPs). It democratises computational materials science by providing a "Zero-Config" workflow.

## Overview
**What**: A robust, type-safe pipeline for automating the generation of MLIPs.
**Why**: Scientific software often suffers from loose data contracts. This project establishes a rigid **Schema-First** foundation to prevent "silent failures" in complex simulations.

## Key Features

-   **One-Shot MD Pipeline (Cycle 02)**:
    -   Automated structure generation (Bulk crystals + Rattle).
    -   Seamless integration with LAMMPS for molecular dynamics.
    -   Automatic input file generation and output parsing.
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
-   **External Tools**: LAMMPS (optional for simulations, mocked for testing)

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

### 3. Run One-Shot MD Pipeline
Execute the Cycle 02 pipeline to generate a structure and run a simulation.
```bash
uv run mlip-auto run-cycle-02 --config config.yaml
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
  cores: 1
logging:
  level: "INFO"
  file_path: "mlip_pipeline.log"
```

## Project Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Schemas (Structure, Config, Job)
├── infrastructure/         # Logging, IO
├── physics/                # Domain Logic (Structure Gen, MD Wrapper)
├── orchestration/          # Workflow Management
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

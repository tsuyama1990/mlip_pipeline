# PyAceMaker: Automated MLIP Pipeline

![Status](https://img.shields.io/badge/Status-Cycle_08_Verified-green)
![Python](https://img.shields.io/badge/Python-3.12%2B-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**PyAceMaker** is an autonomous research system designed to construct State-of-Art Machine Learning Interatomic Potentials (MLIPs). It democratises computational materials science by providing a "Zero-Config" workflow.

## Overview
**What**: A robust, type-safe pipeline for automating the generation of MLIPs.
**Why**: Scientific software often suffers from loose data contracts. This project establishes a rigid **Schema-First** foundation to prevent "silent failures" in complex simulations.

## Key Features

-   **Autonomous Active Learning Loop**:
    -   **State-Aware Orchestration**: Resumable workflow managing Exploration, Selection, Calculation, and Training phases.
    -   **Uncertainty Quantification**: Real-time detection of high-uncertainty configurations during MD (`fix halt`).
    -   **Adaptive Exploration**:
        -   **Smart Policy**: Automatically selects exploration strategies (MD, MC, Static, aKMC) based on material type and history.
        -   **Long-Timescale Exploration**: Integration with **EON** for Adaptive Kinetic Monte Carlo (aKMC) to find rare events and saddle points.
        -   **Advanced Generators**: Creates defects (Vacancies, Interstitials, Antisites) and strained structures to probe phase space boundaries.
    -   **Self-Improvement**: Automatically refines potentials by learning from "confusing" structures.
-   **Production Deployment**:
    -   **Automated Packaging**: One-command deployment (`deploy`) creates a distributable zip file containing the potential, validation report, and rigorous metadata (training size, metrics).
-   **Oracle (DFT Automation)**:
    -   **Self-Healing**: Robust Quantum Espresso wrapper that automatically detects and fixes SCF convergence failures.
    -   **Auto K-Points**: Generates K-point grids dynamically based on physical spacing density.
    -   **Periodic Embedding**: Efficiently extracts atomic clusters from large simulations into vacuum-padded boxes for DFT calculation.
-   **Machine Learning Potential Training**:
    -   **Pacemaker Integration**: Seamless interface for training Atomic Cluster Expansion (ACE) potentials.
    -   **Active Set Selection**: Optimized dataset pruning using D-optimality.
    -   **Delta Learning**: Robust reference potential subtraction (ZBL/LJ).
-   **Validation Framework**:
    -   **Physics Validation**: Automated tests for Phonon stability, Elastic stability, and EOS.
    -   **Reporting**: Generates HTML reports with pass/fail metrics and plots.
-   **Molecular Dynamics Engine**:
    -   Automated "One-Shot" MD pipelines via LAMMPS.
    -   Robust wrapper with input generation, execution management, and trajectory parsing.
-   **Strict Data Validation**: Pydantic-based domain models ensure data integrity.
-   **Configuration Management**: `init`, `check`, and type-safe YAML.
-   **Robust Infrastructure**: Dual-channel logging (Rich console + File).

## Requirements

-   **Python**: 3.12+
-   **Package Manager**: `uv` (recommended)
-   **Simulation Engines**:
    -   **LAMMPS**: `lmp` or `mpirun` (Required for MD).
    -   **Quantum Espresso**: `pw.x` (Required for DFT).
    -   **Pacemaker**: `pace_train`, `pace_activeset` (Required for Training).
    -   **EON**: `eon` (Required for aKMC).

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
```

### 2. Validate Configuration
Check if your configuration file is valid.
```bash
uv run mlip-auto check --config config.yaml
```

### 3. Run Autonomous Loop
Execute the full active learning cycle (Explore -> Detect -> Refine -> Validate).
```bash
uv run mlip-auto run-loop --config config.yaml
```

### 4. Deploy for Production
Package the final potential for distribution.
```bash
uv run mlip-auto deploy --version 1.0.0 --author "Jane Doe" --config config.yaml
# Output: dist/mlip_package_1.0.0.zip
```

### Other Commands
-   **Run One-Shot MD**: `uv run mlip-auto run-one-shot --config config.yaml`
-   **Train Potential**: `uv run mlip-auto train --dataset data.extxyz`
-   **Validate Potential**: `uv run mlip-auto validate --potential potential.yace`

Example `config.yaml` snippet for Cycle 08:
```yaml
eon:
  command: "eon"
  timeout: 86400
orchestrator:
  uncertainty_threshold: 5.0
validation:
  report_path: "validation_report.html"
```

## Project Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Schemas (ProductionManifest, EonConfig, etc.)
├── physics/                # Physics Engines
│   ├── dft/                # Quantum Espresso
│   ├── dynamics/           # LAMMPS & EON (aKMC)
│   ├── training/           # Pacemaker
│   ├── validation/         # Phonons, Elasticity, EOS
│   └── structure_gen/      # Generators & Policy
├── orchestration/          # Workflow (Orchestrator, Phases, State)
├── infrastructure/         # Production, Logging, IO
├── inference/              # Drivers (pace_driver.py)
└── app.py                  # CLI Entry Point
```

## Roadmap

-   **Cycle 01-07**: Core foundations, DFT, Training, Validation, Active Learning, Adaptive Strategy (Completed).
-   **Cycle 08**: Expansion (aKMC) & Production (Deployment) (Completed).

## License

MIT License.

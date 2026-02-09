# MLIP Pipeline

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

**Autonomous Active Learning Pipeline for Machine Learning Interatomic Potentials.**

## Overview

MLIP Pipeline is a modular, automated system for training, validating, and deploying Machine Learning Interatomic Potentials (MLIPs). It orchestrates a complete active learning loop, iteratively refining potentials by exploring atomic configuration space and selecting high-uncertainty structures for first-principles labeling.

This tool solves the problem of manual, error-prone potential fitting by automating:
1.  **Generation**: Creating initial atomic structures.
2.  **Labeling**: Computing energy/forces/stress via DFT (Quantum Espresso/VASP).
3.  **Training**: Fitting potentials (ACE/MACE/GAP via Pacemaker).
4.  **Exploration**: Running MD to find failure cases (Active Learning).
5.  **Validation**: Verifying physical properties (Phonons, Elastic Constants, EOS).

## Features

-   **Modular Architecture**: Plug-and-play components for Generator, Oracle, Trainer, Dynamics, and Validator.
-   **Active Learning**: "Halt & Diagnose" strategy to automatically detect and heal potential failures.
-   **Physical Validation**: Automated calculation of Phonon stability, Bulk/Shear moduli, and Equation of State.
-   **Robust Orchestration**: State management, checkpointing, and error recovery.
-   **Reporting**: Generates HTML reports with learning curves and validation metrics.
-   **Security**: Strict path validation and input sanitization.

## Requirements

-   Python >= 3.12
-   `uv` package manager (recommended) or `pip`.
-   External dependencies (optional but required for full functionality):
    -   LAMMPS (for MD and validation)
    -   Quantum Espresso or VASP (for DFT labeling)
    -   Pacemaker (for training ACE potentials)

## Installation

```bash
git clone <repository_url>
cd mlip-pipeline
uv sync
```

## Usage

### 1. Configuration
Create a `config.yaml` file defining your pipeline.

```yaml
workdir: "./experiment_01"
max_cycles: 5
logging_level: "INFO"

components:
  generator:
    name: "adaptive"
    n_structures: 10
    element: "Si"
    crystal_structure: "diamond"

  oracle:
    name: "qe" # Quantum Espresso
    ecutwfc: 40.0
    ecutrho: 160.0
    kspacing: 0.05

  trainer:
    name: "pacemaker"
    max_num_epochs: 100
    cutoff: 5.0

  dynamics:
    name: "lammps"
    n_steps: 10000
    timestep: 0.001
    uncertainty_threshold: 5.0 # Max Gamma

  validator:
    name: "standard"
    phonon_supercell: [2, 2, 2]
```

### 2. Run Pipeline
Execute the pipeline using the CLI.

```bash
uv run python -m mlip_autopipec run config.yaml
```

### 3. Generate Report
Generate an HTML summary of the run.

```bash
uv run python -m mlip_autopipec report config.yaml
```

## Tutorials

We provide a set of Jupyter Notebooks in the `tutorials/` directory to guide you through a complete scientific workflow (Fe/Pt on MgO).

-   `01_MgO_FePt_Training.ipynb`: Train base potentials for substrate and alloy.
-   `02_Interface_Learning.ipynb`: Learn the metal-oxide interface.
-   `03_Deposition_MD.ipynb`: Simulate Physical Vapor Deposition (PVD).
-   `04_Ordering_aKMC.ipynb`: Explore long-term ordering using aKMC (EON).

To run them:
```bash
# Install dependencies
uv sync

# Run the notebook server
uv run jupyter notebook tutorials/
```

## Architecture

```
src/mlip_autopipec/
├── components/       # Component implementations (Dynamics, Oracle, Trainer, etc.)
│   ├── dynamics/
│   ├── generator/
│   ├── oracle/
│   ├── trainer/
│   └── validator/    # Physical validation (Phonons, Elastic, EOS)
├── core/             # Core logic (Orchestrator, Dataset, State)
│   ├── orchestrator.py
│   ├── dataset.py
│   └── report.py
├── domain_models/    # Pydantic data models
└── interfaces/       # Abstract base classes
```

## Roadmap

-   [ ] Support for MACE and NequIP architectures.
-   [ ] Distributed execution via SLURM.
-   [ ] Advanced sampling strategies (uncertainty quantification).

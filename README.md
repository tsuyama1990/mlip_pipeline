# PYACEMAKER: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)

**PYACEMAKER** is an automated system designed to democratize the creation of Machine Learning Interatomic Potentials (MLIP). By leveraging the Atomic Cluster Expansion (ACE) formalism via Pacemaker, it enables materials scientists to generate "State-of-the-Art" potentials with minimal manual intervention.

From a single configuration file, PYACEMAKER orchestrates the entire lifecycle: generating candidate structures, running DFT calculations (Oracle), training the model (Trainer), and validating its physics (Validator). It features a robust "Active Learning" loop that autonomously detects high-uncertainty regions during Molecular Dynamics (MD) simulations and retrains the potential on-the-fly.

## Features

-   **Zero-Config Automation**: Define your material system in `config.yaml` and let the system handle structure generation, DFT, training, and validation.
-   **Adaptive Structure Generation**: Intelligently explores the configuration space using multiple strategies (Random, Defect, M3GNet) driven by an adaptive policy engine. It automatically switches between cold start (using M3GNet or prototypes) and refinement strategies based on model uncertainty.
-   **Automated DFT (Oracle)**: Integrated wrapper around Quantum Espresso (via ASE) with self-healing retry logic for SCF convergence failures.
-   **Pacemaker Integration (Trainer)**: Seamlessly trains ACE potentials using `pace_train`. Supports active set selection (`pace_activeset`) to filter redundant structures (D-optimality).
-   **Physics Validation (Validator)**: Automatically runs comprehensive physics checks on trained potentials, including Phonon stability (no imaginary frequencies), Equation of State (EOS) bulk modulus, and Elastic constants ($C_{ij}$) stability criteria. Generates HTML reports with plots.
-   **Dynamics Engine Integration**:  Support for running MD simulations via LAMMPS with automated input generation (including hybrid potentials like ACE+ZBL/LJ). Also integrates with **EON** for long-timescale Adaptive Kinetic Monte Carlo (akMC) simulations.
-   **Delta Learning**: Automatically configures physics-based baselines (ZBL/LJ) to ensure core repulsion, allowing the ACE potential to learn only the difference ($E_{ACE} = E_{DFT} - E_{Baseline}$).
-   **Dataset Management**: Efficient handling of large atomic structure datasets (`.pckl.gzip`), fully compatible with Pacemaker. Includes streaming support for large files.
-   **Active Learning Loop**: Uses "Halt & Diagnose" logic to monitor MD simulations. If uncertainty ($\gamma$) spikes, the simulation halts, and the problematic structure is automatically sent for labeling and retraining.

## Requirements

-   **Python 3.11+**
-   **Quantum Espresso** (pw.x) installed and accessible in PATH (or specified in config).
-   **Pacemaker** (ACE training engine)
-   **LAMMPS** (with USER-PACE package installed)
-   **EON** (Optional, for kMC simulations)
-   **uv** (Recommended for dependency management)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```
    *Alternatively, using pip:*
    ```bash
    pip install -e .[dev]
    ```

## Usage

### 1. Configuration
Create a `config.yaml` file. Here is an example with DFT, Trainer, and Generator settings:

```yaml
version: "0.1.0"
project:
  name: "MyMaterial"
  root_dir: "./project_data"

oracle:
  dft:
    code: "quantum_espresso"
    command: "mpirun -np 16 pw.x"
    pseudopotentials:
      Fe: "path/to/Fe.pbe.UPF"
      O: "path/to/O.pbe.UPF"
    kspacing: 0.05
    smearing: 0.02
    max_retries: 3
    parameters:
      system:
        ecutwfc: 80.0
        ecutrho: 320.0
      electrons:
        mixing_beta: 0.5
  mock: false

structure_generator:
  strategy: "adaptive"
  initial_exploration: "m3gnet"
  strain_range: 0.15
  rattle_amplitude: 0.1
  defect_density: 0.01

trainer:
  potential_type: "pace"
  cutoff: 6.0
  order: 3
  basis_size: [15, 5]
  delta_learning: "zbl"
  max_epochs: 500
  batch_size: 100
  mock: false

validator:
  test_set_ratio: 0.1
  phonon_supercell: [3, 3, 3]
  eos_strain: 0.1
  elastic_strain: 0.01

dynamics_engine:
  engine: "lammps" # or "eon"
  timestep: 0.001
  temperature: 300.0
  gamma_threshold: 2.0
  eon:
    executable: "eonclient"

orchestrator:
  max_cycles: 10
```

### 2. Run the Pipeline
Execute the full active learning loop:
```bash
uv run pyacemaker run config.yaml
```

### 3. Tutorial
For a complete interactive walkthrough of the Fe/Pt deposition on MgO scenario, check out the tutorial notebook. This notebook supports both **Mock Mode** (CI) for quick verification and **Production Mode** for real scientific workflows.

```bash
# Run in interactive mode
uv run marimo edit tutorials/UAT_AND_TUTORIAL.py

# Run as a web app
uv run marimo run tutorials/UAT_AND_TUTORIAL.py
```

By default, the tutorial runs in **Mock Mode** (simulated data) for quick verification. To run in **Production Mode** (real DFT/MD), set `CI=false` environment variable (requires Quantum Espresso and LAMMPS installed).

### 4. Oracle (DFT) Usage
You can also use the Oracle module independently in your Python scripts:

```python
from pathlib import Path
from pyacemaker.core.config_loader import load_config
from pyacemaker.modules.oracle import DFTOracle
from pyacemaker.domain_models.models import StructureMetadata
from ase.build import bulk

# Load config
config = load_config(Path("config.yaml"))

# Initialize Oracle
oracle = DFTOracle(config)

# Create a structure
atoms = bulk("Fe", cubic=True)
structure = StructureMetadata(features={"atoms": atoms})

# Compute properties (Energy, Forces, Stress)
results = list(oracle.compute_batch([structure]))

print(f"Energy: {results[0].features['energy']} eV")
```

## Architecture/Structure

```text
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/           # Configuration, Logging, Base Classes
│       ├── domain_models/  # Pydantic Data Models
│       ├── modules/        # High-level Modules (Oracle, Trainer, Generator, etc.)
│       │   ├── oracle.py   # DFTOracle Implementation
│       │   ├── trainer.py  # PacemakerTrainer Implementation
│       │   ├── structure_generator.py # Structure Generator Implementation
│       │   ├── validator.py           # Validator Implementation
│       │   └── dynamics_engine.py     # MD/kMC Engine Implementation
│       ├── oracle/         # DFT Logic Package
│       ├── trainer/        # Trainer Logic Package
│       ├── generator/      # Structure Generation Package (Strategies, Policies)
│       │   ├── strategies.py # Exploration Strategies
│       │   ├── policy.py     # Adaptive Policy Logic
│       │   └── mutations.py  # Atomic Mutations
│       ├── validator/      # Validation Logic Package
│       │   ├── manager.py    # Validator Orchestrator
│       │   ├── physics.py    # Phonon, EOS, Elastic Checks
│       │   └── report.py     # HTML Report Generation
│       ├── dynamics/       # Dynamics Logic Package
│       │   └── kmc.py        # EON Wrapper
│       ├── orchestrator.py # Main Loop
│       └── main.py         # CLI Entry Point
├── tests/                  # Unit & Integration Tests
└── pyproject.toml          # Project Configuration
```

## Roadmap

-   [x] Cycle 01: Core Infrastructure & Orchestrator
-   [x] Cycle 02: Oracle (DFT) & Data Management
-   [x] Cycle 03: Trainer (Pacemaker Integration)
-   [x] Cycle 04: Structure Generation
-   [x] Cycle 05: Dynamics Engine (MD/kMC) & On-the-Fly Learning
-   [x] Cycle 06: Validator & Final Polish

## License

MIT License. See `LICENSE` for details.

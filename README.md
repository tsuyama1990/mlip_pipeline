# PYACEMAKER: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)

**PYACEMAKER** is an automated system designed to democratize the creation of Machine Learning Interatomic Potentials (MLIP). By leveraging the Atomic Cluster Expansion (ACE) formalism via Pacemaker, it enables materials scientists to generate "State-of-the-Art" potentials with minimal manual intervention.

From a single configuration file, PYACEMAKER orchestrates the entire lifecycle: generating candidate structures, running DFT calculations (Oracle), training the model (Trainer), and validating its physics (Validator). It features a robust "Active Learning" loop that autonomously detects high-uncertainty regions during Molecular Dynamics (MD) simulations and retrains the potential on-the-fly.

## Features

-   **Zero-Config Automation**: Define your material system in `config.yaml` and let the system handle structure generation, DFT, training, and validation.
-   **Automated DFT (Oracle)**: Integrated wrapper around Quantum Espresso (via ASE) with self-healing retry logic for SCF convergence failures.
-   **Dataset Management**: Efficient handling of large atomic structure datasets (`.pckl.gzip`), fully compatible with Pacemaker. Includes streaming support for large files.
-   **Active Learning Loop**: Uses "Halt & Diagnose" logic to monitor MD simulations. If uncertainty ($\gamma$) spikes, the simulation halts, and the problematic structure is automatically sent for labeling and retraining.
-   **Physics-Informed Robustness**: Enforces Hybrid Potentials (ACE + ZBL/LJ) to prevent unphysical behavior (e.g., core collapse) in unknown regions.

## Requirements

-   **Python 3.11+**
-   **Quantum Espresso** (pw.x) installed and accessible in PATH (or specified in config).
-   **Pacemaker** (ACE training engine)
-   **LAMMPS** (with USER-PACE package installed)
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
Create a `config.yaml` file. Here is an example with DFT settings:

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
    # Optional: Override default QE parameters
    parameters:
      system:
        ecutwfc: 80.0
        ecutrho: 320.0
      electrons:
        mixing_beta: 0.5
  mock: false

structure_generator:
  strategy: "random"

trainer:
  potential_type: "pace"

orchestrator:
  max_cycles: 10
```

### 2. Run the Pipeline
Execute the full active learning loop:
```bash
uv run pyacemaker run config.yaml
```

### 3. Oracle (DFT) Usage
You can also use the Oracle module independently in your Python scripts:

```python
from pyacemaker.core.config import load_config
from pyacemaker.modules.oracle import DFTOracle
from pyacemaker.domain_models.models import StructureMetadata
from ase.build import bulk

# Load config
config = load_config("config.yaml")

# Initialize Oracle
oracle = DFTOracle(config)

# Create a structure
atoms = bulk("Fe", cubic=True)
structure = StructureMetadata(features={"atoms": atoms})

# Compute properties (Energy, Forces, Stress)
results = oracle.compute_batch([structure])

print(f"Energy: {results[0].features['energy']} eV")
```

### 4. Self-Healing DFT
The DFT module automatically handles common failures:
-   **SCF Convergence**: If a calculation fails to converge, the system retries with adjusted parameters (e.g., reducing `mixing_beta`) up to `max_retries` times.
-   **Fatal Errors**: Syntax errors or missing files cause immediate failure to prevent wasted resources.

## Architecture/Structure

```text
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/           # Configuration, Logging, Base Classes
│       ├── domain_models/  # Pydantic Data Models
│       ├── modules/        # High-level Modules (Oracle, Trainer, etc.)
│       │   └── oracle.py   # DFTOracle Implementation
│       ├── oracle/         # DFT Logic Package
│       │   ├── calculator.py # ASE Calculator Factory
│       │   ├── dataset.py    # Dataset I/O
│       │   └── manager.py    # DFT Execution & Retry Logic
│       ├── orchestrator.py # Main Loop
│       └── main.py         # CLI Entry Point
├── tests/                  # Unit & Integration Tests
└── pyproject.toml          # Project Configuration
```

## Roadmap

-   [x] Cycle 01: Core Infrastructure & Orchestrator
-   [x] Cycle 02: Oracle (DFT) & Data Management
-   [ ] Cycle 03: Trainer (Pacemaker Integration)
-   [ ] Cycle 04: Structure Generation
-   [ ] Cycle 05: Dynamics Engine (MD/kMC)
-   [ ] Cycle 06: Validator & Final Polish

## License

MIT License. See `LICENSE` for details.

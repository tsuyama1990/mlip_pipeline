# PyAceMaker: Automated MLIP Pipeline

![Status](https://img.shields.io/badge/Status-Architecture_Design-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**PyAceMaker** is an autonomous research system designed to construct State-of-the-Art Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) formalism. It democratises computational materials science by providing a "Zero-Config" workflow that handles the complexities of structure generation, DFT calculation recovery, and active learning loops, allowing researchers to focus on physics rather than parameter tuning.

## Key Features

1.  **Zero-Config Automation**: From a simple configuration file (e.g., "Ti-O binary"), the system automatically orchestrates the entire pipeline: Exploration, Labelling, Training, and Validation.
2.  **Physics-Informed Robustness**: Implements **Delta Learning**, enforcing a physical baseline (Lennard-Jones/ZBL) to ensure simulation stability even in deep extrapolation regimes (e.g., high-energy collisions).
3.  **Self-Healing Oracle**: A robust DFT interface (wrapping Quantum Espresso) that automatically detects convergence failures and adjusts mixing parameters or electronic temperature to salvage calculations.
4.  **Active Learning**: Uses an **Adaptive Exploration Policy** to intelligently switch between Molecular Dynamics, Monte Carlo, and Defect Sampling, ensuring the potential learns from "rare events" and critical phase space regions.
5.  **Automated Validation**: Every generated potential is subjected to rigorous physical tests (Phonon Dispersion, Elastic Stability, EOS) before being marked as production-ready.

## Architecture Overview

The system operates as a central Orchestrator managing four specialized workers.

```mermaid
graph TD
    User[User Configuration] -->|config.yaml| Orch[Orchestrator]

    subgraph "Core Logic"
        Orch -->|Request Structures| Gen[Structure Generator]
        Orch -->|Request Simulation| Dyn[Dynamics Engine]
        Orch -->|Request Ground Truth| Oracle[Oracle (DFT)]
        Orch -->|Request Training| Trainer[Trainer (Pacemaker)]
        Orch -->|Request Validation| Valid[Validator]
    end

    subgraph "Data Flow"
        Gen -->|Candidate Structures| Dyn
        Dyn -->|Halted/Uncertain Structures| Oracle
        Oracle -->|Labelled Data (E, F, S)| Trainer
        Trainer -->|Potential (.yace)| Dyn
        Trainer -->|Potential (.yace)| Valid
        Valid -->|Pass/Fail & Report| Orch
    end
```

## Prerequisites

-   **Python**: 3.11 or higher.
-   **Package Manager**: `uv` (recommended) or `pip`.
-   **External Engines**:
    -   [LAMMPS](https://www.lammps.org/) (for MD exploration).
    -   [Quantum Espresso](https://www.quantum-espresso.org/) (for DFT labelling).
    -   [Pacemaker](https://pacemaker.readthedocs.io/) (for ACE training).
    -   [Phonopy](https://phonopy.github.io/phonopy/) (for validation).

## Installation

We recommend using `uv` for fast, reliable dependency management.

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

### Quick Start

1.  **Initialize a Project**:
    Generate a template configuration file.
    ```bash
    mlip-auto init
    ```

2.  **Configure**:
    Edit `config.yaml` to specify your elements and paths to external executables.
    ```yaml
    project_name: "Titanium_Oxide"
    potential:
      elements: ["Ti", "O"]
    dft:
      command: "mpirun -np 16 pw.x"
    ```

3.  **Run the Loop**:
    Start the autonomous active learning cycle.
    ```bash
    mlip-auto run-loop
    ```

4.  **Validate**:
    Manually trigger validation on a generated potential.
    ```bash
    mlip-auto validate potentials/generation_001.yace
    ```

## Development Workflow

This project follows a strict Schema-First design using Pydantic.

### Running Tests
```bash
pytest
```

### Linting & Type Checking
We enforce strict type safety.
```bash
ruff check .
mypy .
```

### Development Cycles
The project is implemented in 8 sequential cycles:
-   **Cycle 01**: Foundation & Core Models
-   **Cycle 02**: Basic Exploration (MD)
-   **Cycle 03**: Oracle (DFT)
-   **Cycle 04**: Training (Pacemaker)
-   **Cycle 05**: Validation Framework
-   **Cycle 06**: Active Learning Loop
-   **Cycle 07**: Adaptive Strategy
-   **Cycle 08**: Expansion (kMC) & Production

## Project Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Schemas (The "Language")
├── orchestration/          # The Brain (State Machine)
├── physics/                # Domain Logic
│   ├── structure_gen/      # Policies, Random packing
│   ├── dft/                # QE Wrapper
│   ├── dynamics/           # LAMMPS/EON wrappers
│   ├── training/           # Pacemaker wrapper
│   └── validation/         # Phonons, Elasticity
└── infrastructure/         # Logging, IO, CLI
```

## License

MIT License. See `LICENSE` for details.

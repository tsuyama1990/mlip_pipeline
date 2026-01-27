# PyAcemaker: Automated MLIP Construction System

![Status](https://img.shields.io/badge/Status-Development-orange)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**PyAcemaker** is a comprehensive, "Zero-Config" software system designed to democratise the creation of "State-of-the-Art" Machine Learning Interatomic Potentials (MLIP). By automating the complex loop of structure generation, Quantum Mechanical calculations (DFT), model training (ACE), and validation, it allows materials scientists to generate robust potentials with minimal human effort and computational cost.

---

## Key Features

-   **Zero-Config Workflow**: Automates the entire pipeline from a single YAML configuration file. No manual scripting required.
-   **Active Learning Efficiency**: Uses uncertainty quantification to select only the most informative structures, reducing DFT costs by >90% compared to random sampling.
-   **Physics-Informed Robustness**: Enforces physical safety (core repulsion) via Hybrid Potentials (ACE + ZBL), preventing simulation crashes in unknown regions.
-   **Self-Healing Oracle**: Automatically detects and corrects DFT convergence failures, ensuring a reliable stream of training data.
-   **Scalable Dynamics**: seamlessly integrates with LAMMPS for MD and EON for Adaptive Kinetic Monte Carlo (aKMC) to explore vast time and length scales.

## Architecture Overview

PyAcemaker orchestrates a set of specialized modules to drive the Active Learning Cycle.

```mermaid
graph TD
    User[User Config (YAML)] --> Orch{Orchestrator}
    Orch -->|1. Explore| Gen[Structure Generator]
    Orch -->|1. Explore| Dyn[Dynamics Engine]

    Dyn -->|Halt on High Uncertainty| Orch
    Gen -->|Candidate Structures| Orch

    Orch -->|2. Select| Trainer[Trainer / Active Set]
    Trainer -->|Selected Candidates| Oracle[Oracle (DFT)]

    Oracle -->|3. Compute (Energy/Forces)| DB[(Database)]
    DB --> Trainer

    Trainer -->|4. Train| Pot[Potential (YACE)]
    Pot -->|5. Validate| Val[Validator]

    Val -- Pass --> Orch
    Val -- Fail --> Gen

    Dyn -.->|Uses| Pot
```

## Prerequisites

-   **Python 3.11+**
-   **uv** (Modern Python package manager)
-   **Quantum Espresso** (`pw.x`) - For DFT calculations.
-   **LAMMPS** (`lmp`) - For MD simulations (must be compiled with `USER-PACE`).
-   **Pacemaker** - For training ACE potentials.

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies**:
    We use `uv` for fast and reliable dependency management.
    ```bash
    uv sync
    ```

3.  **Configure Environment**:
    Copy the example configuration and adjust paths to your local binaries.
    ```bash
    cp config.example.yaml config.yaml
    # Edit config.yaml to set paths for pw.x, lmp, etc.
    ```

## Usage

### Running the Active Learning Loop
To start an autonomous potential generation campaign:

```bash
uv run mlip-auto run-loop --config config.yaml
```

### Validating a Potential
To run the physics validation suite on an existing potential:

```bash
uv run mlip-auto validate --potential potentials/my_potential.yace
```

## Development Workflow

This project follows the AC-CDD (Architect-Coder-Cycle-Driven Development) methodology.

### Running Tests
```bash
uv run pytest
```

### Linting and Formatting
We enforce strict code quality using `ruff` and `mypy`.
```bash
uv run ruff check .
uv run mypy .
```

## Project Structure

```ascii
mlip_autopipec/
├── config/              # Pydantic schemas for configuration
├── orchestration/       # Main loop logic (The Brain)
├── generator/           # Structure generation (The Explorer)
├── dft/                 # Quantum Espresso wrapper (The Oracle)
├── trainer/             # Pacemaker wrapper (The Learner)
├── dynamics/            # LAMMPS/EON interface (The Runner)
├── validation/          # Physics checks (The Judge)
└── app.py               # CLI Entry point
```

## License

This project is licensed under the MIT License.

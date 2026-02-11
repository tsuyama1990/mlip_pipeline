# PyAceMaker: Automated Machine Learning Interatomic Potential Construction

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**PyAceMaker** (`mlip_autopipec`) is a "Zero-Config" autonomous system designed to democratise the creation of State-of-the-Art Machine Learning Interatomic Potentials (MLIPs). By orchestrating `Pacemaker`, `QuantumEspresso`, `LAMMPS`, and `EON`, it allows researchers to generate robust, physics-informed potentials for complex materials without writing a single line of code.

## Key Features

-   **Zero-Config Workflow**: A single `config.yaml` drives the entire lifecycle from initial structure generation to final validation.
-   **Active Learning Loop**: Autonomous "Explore -> Halt -> Label -> Train" cycle using D-Optimality (MaxVol) to minimize expensive DFT calculations.
-   **Physics-Informed Robustness**: Enforces Delta Learning against physical baselines (LJ/ZBL) and uses Hybrid Potentials in production to prevent non-physical fusion.
-   **Self-Healing Oracle**: Automatically detects and fixes DFT convergence errors (SCF params, mixing beta).
-   **Time-Scale Bridging**: Seamlessly integrates MD (nanoseconds) with Adaptive Kinetic Monte Carlo (seconds/hours) to capture rare events.

## Architecture Overview

PyAceMaker follows a Hub-and-Spoke architecture where the **Orchestrator** manages specialized components.

```mermaid
graph TD
    User[User / Config.yaml] --> Orch[Orchestrator]

    subgraph "Core Loop"
        Orch -->|1. Request Structures| Gen[Structure Generator]
        Orch -->|2. Compute Properties| Oracle[Oracle (DFT)]
        Orch -->|3. Train Model| Trainer[Trainer (Pacemaker)]
        Orch -->|4. Run Simulation| Dyn[Dynamics Engine]
    end

    subgraph "Quality Gate"
        Orch -->|5. Verify| Val[Validator]
    end

    Dyn -.->|Uncertainty Halt| Gen
```

## Prerequisites

-   **Python 3.12+**
-   **uv** (Recommended package manager)
-   **Docker / Singularity** (Optional, for running external binaries like QE/LAMMPS)
-   **External Binaries** (if running in Real Mode):
    -   `pw.x` (Quantum Espresso)
    -   `lmp_mpi` (LAMMPS with USER-PACE)
    -   `pace_train` (Pacemaker)
    -   `eonclient` (EON)

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/mlip_autopipec.git
    cd mlip_autopipec
    ```

2.  **Install dependencies:**
    ```bash
    uv sync
    ```

3.  **Configure environment:**
    ```bash
    cp .env.example .env
    # Edit .env to point to your binary paths if needed
    ```

## Usage

### Quick Start (CI/Mock Mode)
To verify the installation without external binaries:

```bash
# Initialize a new project directory
mkdir my_project
cd my_project

# Run the pipeline in CI mode (uses Mock components)
export CI=true
uv run mlip-runner run ../examples/config_ci.yaml
```

### Production Run
For a real scientific run:

```bash
uv run mlip-runner run config.yaml
```

### Monitoring
The system produces a `workflow_state.json` and a detailed log file. You can also view the validation report:
```bash
open active_learning/iter_XXX/validation_report.html
```

## Development Workflow

We follow a strict quality assurance process.

**Run Tests:**
```bash
uv run pytest
```

**Run Linters:**
```bash
uv run ruff check .
uv run mypy .
```

**Project Structure:**
```ascii
src/mlip_autopipec/
├── components/      # Worker modules (Generator, Oracle, Trainer, etc.)
├── core/            # Logic (Orchestrator, State Manager)
├── domain_models/   # Pydantic data models
└── main.py          # CLI Entry point

dev_documents/       # Architectural specifications (Cycle 01-08)
tests/               # Unit and Integration tests
tutorials/           # Jupyter notebooks
```

## License

MIT License. See [LICENSE](LICENSE) for details.

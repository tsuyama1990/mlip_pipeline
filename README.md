# PyAcemaker: Automated MLIP Construction System

![Status](https://img.shields.io/badge/Status-Development-orange)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**PyAcemaker** is an autonomous system designed to democratize the creation of "State-of-the-Art" Machine Learning Interatomic Potentials (MLIPs). By orchestrating **Pacemaker** (Atomic Cluster Expansion), **Quantum Espresso** (DFT), and **LAMMPS** (MD), it enables researchers to generate high-fidelity potentials with minimal human intervention.

Unlike traditional workflows that require manual iteration and deep expertise, PyAcemaker operates as a "Self-Driving" research assistant. It explores chemical space, detects gaps in its knowledge, performs necessary first-principles calculations, and refines the potential automatically—all while ensuring physical robustness through hybrid potential architectures.

---

## Key Features

*   **Zero-Config Workflow**: Define your material and constraints in a single `config.yaml`. The system handles the rest, from initial structure generation to final deployment.
*   **Active Learning Loop**: Utilizes uncertainty quantification (extrapolation grade $\gamma$) to smartly sample the phase space. It only performs expensive DFT calculations on structures that maximize information gain (D-Optimality), reducing computational costs by >90%.
*   **Physics-Informed Robustness**: Implements a **Hybrid Potential** strategy (`pair_style hybrid/overlay`), combining the ML model with a physical baseline (ZBL/LJ). This prevents unphysical atomic overlap and simulation crashes in unexplored high-energy regions.
*   **Self-Healing Oracle**: The DFT module automatically detects SCF convergence failures and retries with adjusted parameters (mixing beta, smearing), ensuring robust dataset generation without manual baby-sitting.
*   **Comprehensive Validation**: Every generated potential undergoes a rigorous battery of physical tests—Phonon stability, Elastic constants, and Equation of State (EOS)—before being marked as production-ready.

---

## Architecture Overview

PyAcemaker follows a modular architecture managed by a central **Orchestrator**.

```mermaid
graph TD
    User[User Configuration] -->|Config| Orch(Orchestrator)

    subgraph "Active Learning Cycle"
        Orchestrator -->|Deploy Potential| Dyn[Dynamics Engine<br/>LAMMPS / EON]
        Dyn -->|High Uncertainty Halt| Cands[Candidate Structures]
        Cands -->|Filter (D-Optimality)| Select[Selected Candidates]
        Select -->|Periodic Embedding| Oracle[Oracle<br/>Quantum Espresso]
        Oracle -->|Forces & Energies| Train[Trainer<br/>Pacemaker]
        Train -->|New Potential| Valid[Validator<br/>Phonons/EOS/Elasticity]
        Valid -->|Pass/Fail| Orch
    end

    subgraph "Adaptive Strategy"
        Gen[Structure Generator] -->|Policy: Temp/Pressure/Defects| Dyn
        Orchestrator -->|Feedback| Gen
    end
```

### Core Components
*   **Orchestrator**: The brain of the operation. Manages state, error recovery, and the workflow lifecycle.
*   **Structure Generator**: Proactively engineers defects, strains, and surfaces to seed the exploration.
*   **Dynamics Engine**: Runs MD and kMC simulations. Features "On-the-Fly" monitoring to halt simulations immediately when they enter unknown territory.
*   **Oracle**: Executes DFT calculations with self-correction capabilities.
*   **Trainer**: Wraps Pacemaker to train ACE potentials and optimize the active set.
*   **Validator**: Ensures the potential is physically sound (no imaginary phonons, stable elastic moduli).

---

## Prerequisites

To run the full pipeline, the following external tools are required:

*   **Python 3.11+**
*   **uv** (for fast python dependency management)
*   **Quantum Espresso** (`pw.x`) - For DFT calculations.
*   **LAMMPS** (`lmp_mpi` or `lmp_serial`) - Must be compiled with `USER-PACE` package.
*   **Pacemaker** - For training ACE potentials.
*   **EON** (Optional) - For kMC simulations.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    We use `uv` for dependency management.
    ```bash
    uv sync
    ```

3.  **Environment Setup**
    Copy the example environment file and configure your paths.
    ```bash
    cp .env.example .env
    # Edit .env to point to your local pw.x and lmp executables
    ```

## Usage

### Quick Start

1.  **Prepare Configuration**
    Create a `config.yaml` file describing your system (e.g., Titanium Oxide).
    ```yaml
    project_name: "TiO2_Pilot"
    workflow:
      max_cycles: 5
    dft:
      command: "mpirun -np 4 pw.x"
      pseudopotentials: "./pseudos"
    ```

2.  **Run the System**
    ```bash
    uv run pyacemaker --config config.yaml
    ```

3.  **Monitor Progress**
    Check the `workspace/logs/` directory for detailed execution logs.
    Real-time status is updated in `workspace/status.json`.

## Development Workflow

This project strictly adheres to the **AC-CDD (Architecturally Constrained - Cycle Driven Development)** methodology.

### Running Tests
We use `pytest` for unit and integration tests.
```bash
uv run pytest
```

### Code Quality
Strict linting is enforced via `ruff` and type checking via `mypy`.
```bash
uv run ruff check .
uv run mypy .
```

### Project Structure
```ascii
src/mlip_autopipec/
├── config/             # Pydantic configuration models
├── orchestration/      # Workflow logic and state management
├── dft/                # Quantum Espresso interface (Oracle)
├── trainer/            # Pacemaker interface (Trainer)
├── dynamics/           # LAMMPS/EON interface (Explorer)
├── generator/          # Structure generation (Defects/Strain)
└── validation/         # Physical validation suite
dev_documents/          # Architecture and Cycle specifications
tests/                  # Comprehensive test suite
```

## License

MIT License

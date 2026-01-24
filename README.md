# MLIP Auto PiPEC: Automated Machine Learning Interatomic Potential Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**MLIP Auto PiPEC** is a fully automated, active-learning based system for generating state-of-the-art Machine Learning Interatomic Potentials (MLIPs). It democratizes access to high-accuracy atomic simulations by replacing manual, expert-driven workflows with a robust, "Zero-Config" autonomous pipeline that handles everything from initial structure generation to DFT calculations, training, and validation.

## Key Features

*   **Zero-Config Workflow**: Go from chemical composition to a production-ready potential with a single `config.yaml`.
*   **Active Learning**: Drastically reduces DFT costs by intelligently selecting only the most informative structures (D-Optimality) to label.
*   **Physics-Informed Robustness**: Guarantees simulation stability in unexplored regions by enforcing core repulsion via Delta Learning (ACE + ZBL/LJ).
*   **Self-Healing Oracle**: Automatically recovers from Quantum Espresso/DFT convergence failures by adjusting mixing parameters and algorithms.
*   **Scalable Architecture**: Seamlessly transitions from local prototyping to HPC production runs using a modular, container-ready design.

## Architecture Overview

The system operates as a closed loop where the **Orchestrator** manages the flow of data between the **Generator** (Explorer), **Oracle** (Labeler), **Trainer** (Learner), and **Inference Engine** (Validator).

```mermaid
graph TD
    User[User / Config] -->|Initializes| Orch[Orchestrator]
    Orch -->|Manages| State[Workflow State & DB]

    subgraph "Cycle 1: Exploration"
        Orch -->|Request| Gen[Structure Generator]
        Gen -->|MD/MC/Defects| Candidates[Candidate Structures]
    end

    subgraph "Cycle 2: Oracle"
        Orch -->|Select & Embed| DFT[DFT Oracle (QE/VASP)]
        DFT -->|Forces & Energy| Dataset[Labeled Dataset]
    end

    subgraph "Cycle 3: Training"
        Orch -->|Train| Trainer[Pacemaker Trainer]
        Dataset --> Trainer
        Trainer -->|Produces| Pot[Potential.yace]
    end

    subgraph "Cycle 4: Inference & AL"
        Orch -->|Deploy| MD[Dynamics Engine (LAMMPS/EON)]
        Pot --> MD
        MD -->|Uncertainty (Gamma)| Watchdog[Watchdog Monitor]
        Watchdog -->|High Uncertainty| Halt[Halt & Recovery]
        Halt -->|New Candidates| Gen
    end

    classDef module fill:#f9f,stroke:#333,stroke-width:2px;
    class Orch,Gen,DFT,Trainer,MD module;
```

## Prerequisites

*   **Python**: 3.11 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **External Engines**:
    *   **Quantum Espresso** (`pw.x`) for DFT calculations
    *   **LAMMPS** (`lmp_serial` or `lmp_mpi`) for MD simulations
    *   **Pacemaker** (`pace_train`, `pace_activeset`) for training ACE potentials

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/mlip-autopipec.git
    cd mlip-autopipec
    ```

2.  **Install Dependencies (using uv)**
    ```bash
    uv sync
    ```
    *Alternatively, using pip:*
    ```bash
    pip install .
    ```

3.  **Environment Setup**
    Ensure external binaries are in your `$PATH` or configured in `config.yaml`.

## Usage

### Quick Start

1.  **Initialize a Project**
    Create a `config.yaml` defining your system (e.g., Aluminum):
    ```yaml
    system:
      elements: ["Al"]
    dft:
      command: "mpirun -np 4 pw.x"
    ```

2.  **Run the Pipeline**
    Launch the autonomous loop:
    ```bash
    mlip-auto run config.yaml
    ```

3.  **Monitor Progress**
    The system will output logs to the console and save the workflow state to `mlip_state.json`. You can inspect the database:
    ```bash
    mlip-auto analyze mlip.db
    ```

## Development Workflow

We follow a strict 8-Cycle development roadmap.

### Running Tests
Unit and integration tests are managed by `pytest`.
```bash
pytest
```

### Code Quality
We enforce strict typing and style guidelines.
```bash
ruff check .
mypy .
```

## Project Structure

```ascii
mlip_autopipec/
├── app.py                      # CLI Entry Point
├── config/                     # Configuration Schemas
├── data_models/                # Pydantic Models (Atoms, State)
├── generator/                  # Structure Generation (Adaptive Policy)
├── dft/                        # DFT Oracle & Recovery
├── training/                   # Pacemaker Wrapper
├── inference/                  # LAMMPS & EON Interfaces
├── orchestration/              # Active Learning Loop
└── utils/                      # Logging & Helpers
dev_documents/                  # Architecture & Specifications
tests/                          # Test Suite
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# PYACEMAKER: Automated Active Learning for ML Potentials

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12+-blue)

**PYACEMAKER** is an autonomous "Self-Driving Laboratory" for creating state-of-the-art Machine Learning Interatomic Potentials (MLIP). By orchestrating Density Functional Theory (DFT) calculations, Active Learning (AL), and Molecular Dynamics (MD) validation into a single "Zero-Config" workflow, it allows researchers to generate robust potentials for complex alloys and interfaces without needing to be experts in data science or ML engineering.

## Key Features

*   **Zero-Config Workflow**: Define your elements and goals in a single YAML file. The system handles hyperparameters, convergence, and error recovery automatically.
*   **Active Learning with Uncertainty**: Utilizes the extrapolation grade $\gamma$ to identify and label "dangerous" structures on the fly, preventing unphysical behavior in simulations.
*   **Physics-Informed Robustness**: Enforces "Core Repulsion" using ZBL/LJ baselines, ensuring simulations never crash due to atomic overlap.
*   **Scale-Up Capabilities**: Seamlessly bridges the gap between nanosecond MD and second-scale Adaptive Kinetic Monte Carlo (aKMC) via EON integration.
*   **Self-Healing Oracle**: Automated DFT interface (Quantum Espresso) that detects convergence failures and adjusts parameters dynamically to recover.

## Architecture Overview

The system follows a Hub-and-Spoke architecture centered around an Orchestrator that manages the Active Learning Cycle.

```mermaid
graph TD
    User((User)) -->|config.yaml| Orch[Orchestrator]

    subgraph "Cycle: Active Learning Loop"
        Orch -->|1. Request Candidates| SG[Structure Generator]
        SG -->|2. Structures| Orch

        Orch -->|3. Submit Candidates| Oracle[Oracle (DFT)]
        Oracle -->|4. Labeled Data (E, F, S)| DB[(Training Dataset)]

        DB -->|5. Load Data| Trainer[Trainer (Pacemaker)]
        Trainer -->|6. Fit Potential| Pot[potential.yace]

        Pot -->|7. Update| DE[Dynamics Engine (MD/kMC)]
        DE -->|8. Run Sim & Monitor| DE
        DE -- "Halt! (High Uncertainty)" --> SG
    end

    Pot -->|9. Validate| Val[Validator]
    Val -->|Pass| Prod[Production Ready]
    Val -->|Fail| Orch
```

## Prerequisites

*   **Python**: 3.12 or higher.
*   **Package Manager**: `uv` (recommended) or `pip`.
*   **External Engines**:
    *   **Quantum Espresso** (`pw.x`): For DFT ground truth generation.
    *   **LAMMPS** (`lmp_serial` or `lmp_mpi`): For MD simulations.
    *   **Pacemaker**: For ACE potential training.
    *   **EON**: (Optional) For kMC simulations.

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Initialize Environment (using uv)**
    ```bash
    uv sync
    ```

3.  **Configure Environment**
    Copy the example configuration:
    ```bash
    cp config.example.yaml config.yaml
    # Edit config.yaml with your specific project details
    ```

## Usage

### Quick Start (Mock Mode)
To verify the installation without running heavy physics calculations, run the system in Mock Mode:

```bash
# Ensure config.yaml has type: "mock" for all components
python -m mlip_autopipec.main run config.yaml
```

### Production Run
1.  Set `type: quantum_espresso` and `type: lammps` in `config.yaml`.
2.  Start the active learning loop:
    ```bash
    python -m mlip_autopipec.main run config.yaml
    ```

### Validation
To run the validation suite on a trained potential:
```bash
python -m mlip_autopipec.main validate potentials/generation_005.yace
```

## Development Workflow

This project adheres to the **AC-CDD (Architecturally Constrained Cycle-Driven Development)** methodology.

*   **Linting**: strict adherence to rules is enforced.
    ```bash
    uv run ruff check .
    uv run mypy .
    ```

*   **Testing**:
    ```bash
    uv run pytest
    ```

*   **Cycle-Based Implementation**:
    Development proceeds in 6 cycles:
    1.  **Cycle 01**: Foundation & Mocks
    2.  **Cycle 02**: Oracle (DFT) & Structure Generator
    3.  **Cycle 03**: Trainer (Pacemaker) Integration
    4.  **Cycle 04**: Dynamics Engine (MD) & Uncertainty
    5.  **Cycle 05**: Scale-Up (kMC & Adaptive Policy)
    6.  **Cycle 06**: Validation & Production Readiness

## Project Structure

```
PYACEMAKER/
├── pyproject.toml              # Dependencies & Config
├── src/mlip_autopipec/         # Source Code
│   ├── orchestrator/           # Core Logic
│   ├── oracle/                 # DFT Interface
│   ├── trainer/                # Pacemaker Interface
│   ├── dynamics/               # LAMMPS/EON Interface
│   └── validator/              # Quality Assurance
├── tests/                      # Test Suite
├── dev_documents/              # Design Specs & Architecture
└── tutorials/                  # User Acceptance Tests (Notebooks)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

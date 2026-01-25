# PyAcemaker: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PyAcemaker** is an autonomous system for constructing and operating Machine Learning Interatomic Potentials (MLIPs). Built around the Pacemaker engine, it democratises access to state-of-the-art atomic simulations by providing a "Zero-Config" workflow that automates the complex cycle of structure exploration, DFT calculation, training, and validation.

## Key Features

1.  **Zero-Config Automation**: Operate the entire Active Learning loop from a single YAML file. No Python scripting required.
2.  **Data Efficiency**: Uses Active Learning and "Local D-Optimality" to select only the most informative structures, reducing DFT costs by 90% compared to random sampling.
3.  **Physical Robustness**: Enforces physics-informed baselines (Delta Learning) and performs rigorous validation (Phonons, Elasticity, EOS) to ensure potentials are safe and stable.
4.  **Advanced Exploration**: Integrates Molecular Dynamics (LAMMPS) and Adaptive Kinetic Monte Carlo (EON) to explore both thermal vibrations and rare diffusion events.
5.  **Self-Healing Oracle**: Automated DFT execution (Quantum Espresso) with built-in error recovery for convergence failures.

## Architecture Overview

The system is composed of six independent modules orchestrated by a central controller.

```mermaid
graph TD
    User[User Configuration] --> Orch[Orchestrator]
    Orch --> SG[Structure Generator]
    Orch --> DE[Dynamics Engine<br/>(LAMMPS/EON)]
    Orch --> Oracle[Oracle<br/>(DFT/QE)]
    Orch --> Trainer[Trainer<br/>(Pacemaker)]
    Orch --> Valid[Validator]

    subgraph "Active Learning Loop"
        SG -->|Candidates| Oracle
        DE -->|Halted Structures| Oracle
        Oracle -->|Labelled Data| Trainer
        Trainer -->|Potential.yace| DE
        Trainer -->|Potential.yace| Valid
    end

    Valid -->|Pass/Fail| Orch
    DE -->|Uncertainty Metric| Orch
```

## Prerequisites

*   **Python**: 3.11 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **External Tools**:
    *   Quantum Espresso (`pw.x`)
    *   LAMMPS (`lmp`) with USER-PACE package
    *   Pacemaker
    *   EON (for kMC features)

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    We recommend using `uv` for fast dependency management.
    ```bash
    uv sync
    source .venv/bin/activate
    ```
    Alternatively:
    ```bash
    pip install .
    ```

3.  **Prepare Environment**
    Copy the example configuration.
    ```bash
    cp config.example.yaml config.yaml
    ```

## Usage

### Quick Start
To run the full active learning loop:

```bash
mlip-auto run config.yaml
```

This command will:
1.  Initialize the project directories.
2.  Start the Orchestrator.
3.  Loop through Exploration, Selection, Calculation, and Training.
4.  Generate a `report.html` dashboard.

### Monitoring
You can monitor the progress by viewing the dashboard:
```bash
open active_learning/report.html
```

### Manual Validation
To validate an existing potential against physical criteria:
```bash
mlip-auto validate potential.yace --config config.yaml
```

## Development Workflow

This project follows a strict cycle-based development plan.

*   **Linting**: Ensure code quality before committing.
    ```bash
    ruff check src/
    mypy src/
    ```
*   **Testing**: Run the test suite.
    ```bash
    pytest tests/
    ```

## Project Structure

```ascii
src/mlip_autopipec/
├── app.py                  # CLI Entry Point
├── orchestrator/           # Central Control Logic
├── config/                 # Pydantic Schemas
├── dft/                    # Oracle (Quantum Espresso)
├── training/               # Trainer (Pacemaker)
├── inference/              # Dynamics (LAMMPS/EON)
├── generator/              # Structure Generator
├── validation/             # Quality Assurance
└── utils/                  # Logging and Helpers
```

## License

MIT License

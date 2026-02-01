# PYACEMAKER

**Automated Machine Learning Interatomic Potential Construction System**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**PYACEMAKER** democratizes the creation of state-of-the-art Machine Learning Interatomic Potentials (MLIPs). By leveraging the Atomic Cluster Expansion (ACE) framework and an intelligent Orchestrator, it allows materials scientists to go from "Chemical Composition" to "Production-Ready Potential" with zero coding required.

## Key Features

*   **Zero-Config Workflow**: Define your system in a single `config.yaml` and let the Orchestrator handle the rest. No complex Python scripting needed.
*   **Active Learning**: Drastically reduces DFT costs (by ~90%) by intelligently selecting only the most informative structures using uncertainty quantification ($\gamma$).
*   **Physics-Informed Robustness**: Automatically constructs hybrid potentials (ACE + ZBL/LJ) to ensure simulations never crash due to non-physical atomic overlaps, even in high-energy regimes.
*   **Self-Healing Oracle**: The DFT interface automatically detects convergence failures (SCF errors) and adjusts parameters to recover, minimizing manual intervention.
*   **Automated Validation**: Every generated potential is rigorously tested for Phonon stability, Elastic properties, and Equation of State before being released.

## Architecture Overview

PYACEMAKER uses a modular architecture where a central Orchestrator manages the flow of data between specialized agents.

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]

    subgraph "Active Learning Loop"
        Orch -->|1. Explore| Gen[Structure Generator]
        Gen -->|2. Sample| Dyn[Dynamics Engine (LAMMPS/EON)]
        Dyn -->|3. Halt on Uncertainty| Select[Selection Logic]
        Select -->|4. Label| Oracle[Oracle (DFT/QE)]
        Oracle -->|5. Train| Trainer[Trainer (Pacemaker)]
        Trainer -->|6. Update| Dyn
    end

    Trainer -->|7. Verify| Val[Validator]
    Val -->|8. Report| Report[HTML Report]
```

## Prerequisites

To run PYACEMAKER in "Real Mode", you need the following tools installed and accessible in your `$PATH`:

*   **Python 3.12+**
*   **Quantum Espresso** (`pw.x`)
*   **LAMMPS** (`lmp` or `lmp_serial`) with `USER-PACE` package installed.
*   **Pacemaker** (`pace_train`, `pace_activeset`)
*   **uv** (recommended for Python dependency management)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-autopipec.git
    cd mlip-autopipec
    ```

2.  **Install dependencies using `uv`**:
    ```bash
    uv sync
    ```
    *Alternatively, use pip:*
    ```bash
    pip install .
    ```

3.  **Setup Environment**:
    Copy the example environment file (if available) or ensure your physics codes are in the path.
    ```bash
    export PATH=$PATH:/path/to/quantum-espresso/bin:/path/to/lammps/bin
    ```

## Usage

### Quick Start

1.  **Initialize a new project**:
    Create a working directory and a configuration file.
    ```bash
    mkdir my_silicon_project
    cd my_silicon_project
    # Create a config.yaml (see examples/config.yaml)
    ```

2.  **Run the Orchestrator**:
    ```bash
    uv run python -m mlip_autopipec.main config.yaml
    ```

3.  **Monitor Progress**:
    The system will create an `active_learning/` directory. You can track the progress by checking the logs or the generated validation reports.

## Development Workflow

We use a strict cycle-based development process.

### Running Tests
```bash
uv run pytest
```

### Linting & Code Quality
We enforce strict typing and style rules.
```bash
uv run ruff check .
uv run mypy .
```

## Project Structure

```
mlip-autopipec/
├── src/                    # Source code
│   └── mlip_autopipec/
│       ├── orchestration/  # Main logic
│       ├── physics/        # MD/DFT/Gen interfaces
│       └── validation/     # QA tools
├── tests/                  # Unit and Integration tests
├── dev_documents/          # Specs and Design docs
├── tutorials/              # Jupyter notebooks
└── pyproject.toml          # Project configuration
```

## License

MIT License. See `LICENSE` file for details.

# PyAcemaker: Automated Machine Learning Interatomic Potential Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PyAcemaker** is a "Zero-Config" automation system designed to democratise the creation of State-of-the-Art (SOTA) Machine Learning Interatomic Potentials (MLIPs). Built around the powerful `Pacemaker` (Atomic Cluster Expansion) engine, it manages the entire active learning lifecycle—from initial structure generation and DFT ground-truth calculation to potential fitting and validation—without requiring the user to write a single line of code.

## Key Features

*   **Zero-Config Workflow**: Define your material system (e.g., "Ti-O") in a single YAML file. The system handles exploration, training, and validation autonomously.
*   **Data Efficiency**: Utilizes **Active Learning** (D-Optimality and Uncertainty Quantification) to build high-fidelity potentials with 1/10th the DFT cost of traditional random sampling.
*   **Physics-Informed Robustness**: Implements **Delta Learning** with ZBL/LJ baselines, ensuring simulations never crash due to unphysical forces (core overlap) in extrapolation regions.
*   **Scalable Dynamics**: Seamlessly integrates **Molecular Dynamics (LAMMPS)** for rapid exploration and **Kinetic Monte Carlo (EON)** for long-timescale rare event sampling.
*   **Self-Healing Oracle**: Automatically detects and corrects DFT convergence failures, managing complex `Quantum Espresso` calculations robustly.

## Architecture Overview

The system operates on a Hub-and-Spoke architecture managed by a central Orchestrator.

```mermaid
graph TD
    User[User] -->|Config.yaml| Orch[Orchestrator]

    subgraph "Core Loop"
        Orch -->|Request Structures| Gen[Structure Generator]
        Gen -->|Candidate Structures| Orch

        Orch -->|Submit Jobs| Oracle[Oracle (DFT)]
        Oracle -->|Forces & Energies| DB[(Database)]

        DB -->|Training Set| Trainer[Trainer (Pacemaker)]
        Trainer -->|Potential.yace| Dyn[Dynamics Engine]

        Dyn -->|Run MD/kMC| Dyn
        Dyn -- Halted (High Uncertainty) --> Orch
    end

    subgraph "Validation"
        Trainer -->|Candidate Potential| Val[Validator]
        Val -->|Pass/Fail| Orch
    end

    Dyn -->|Final Model| Deploy[Production]
```

## Prerequisites

*   **Python**: Version 3.11 or higher.
*   **Package Manager**: `uv` (recommended) or `pip`.
*   **External Engines**:
    *   `Quantum Espresso` (pw.x) for DFT.
    *   `LAMMPS` (with USER-PACE package) for MD.
    *   `Pacemaker` (Python package) for training.

## Installation & Setup

We recommend using `uv` for fast and reliable dependency management.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/mlip_autopipec.git
    cd mlip_autopipec
    ```

2.  **Install Dependencies**
    ```bash
    uv sync
    ```
    This will create a virtual environment and install all required packages including development tools.

3.  **Environment Setup**
    Copy the example configuration to set up your environment (paths to executables).
    ```bash
    cp .env.example .env
    # Edit .env to point to your pw.x and lmp executables
    ```

## Usage

### Quick Start
To generate a potential for a simple system (e.g., Aluminum):

1.  Create a configuration file `input.yaml`:
    ```yaml
    project:
      name: "Al_Basic"
      elements: ["Al"]

    dft:
      pseudopotential_dir: "/path/to/sssp/"
    ```

2.  Run the pipeline:
    ```bash
    uv run mlip-auto start input.yaml
    ```

### Validation Only
To validate an existing potential:
```bash
uv run mlip-auto validate --potential potential.yace --structure Al.cif
```

## Development Workflow

This project follows the AC-CDD (Architectural-Centric Cycle-Driven Development) methodology.

### Running Tests
We use `pytest` with coverage tracking.
```bash
uv run pytest
```

### Linting & Formatting
Strict code quality is enforced via `ruff` and `mypy`.
```bash
# Check for errors
uv run ruff check .

# Auto-fix simple errors
uv run ruff check --fix .

# Type checking
uv run mypy .
```

## Project Structure

```
mlip_autopipec/
├── dev_documents/          # Documentation & Specifications
│   └── system_prompts/     # Cycle Definitions (AC-CDD)
├── src/
│   └── mlip_autopipec/
│       ├── orchestrator/   # Main Logic
│       ├── generator/      # Structure Creation
│       ├── dft/            # DFT Interface
│       ├── trainer/        # Pacemaker Interface
│       ├── dynamics/       # LAMMPS/EON Interface
│       └── validator/      # Physics Validation
├── tests/                  # Unit & Integration Tests
├── pyproject.toml          # Config & Dependencies
└── README.md
```

## License

This project is licensed under the MIT License.

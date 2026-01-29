# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

**PYACEMAKER** is a fully automated, "Zero-Config" system for constructing and operating State-of-the-Art Machine Learning Interatomic Potentials (MLIP) using the Pacemaker (ACE) engine. It democratizes computational materials science by enabling researchers to generate robust, physics-informed potentials without needing deep expertise in machine learning or DFT.

## Key Features

*   **Zero-Config Workflow**: Define your material in a single YAML file, and the system handles structure generation, DFT calculations, training, and validation autonomously.
*   **Active Learning Loop**: Utilizes an uncertainty-driven "On-the-Fly" (OTF) learning cycle to sample rare events and high-energy configurations, achieving high accuracy with minimal DFT cost.
*   **Physics-Informed Robustness**: Implements Hybrid Potentials (ACE + ZBL/LJ) to ensure physical behavior in core regions and prevent simulation crashes.
*   **Adaptive Exploration**: Dynamically adjusts sampling strategies (MD vs. MC, temperature schedules) based on material properties (e.g., metals vs. insulators).
*   **Automated Validation**: Built-in quality assurance suite checking Phonon stability, Elastic constants, and Equations of State.

## Architecture Overview

The system is orchestrated by a central controller that manages a closed-loop cycle of Exploration, Oracle labeling, Training, and Validation.

```mermaid
graph TD
    User[User Config] --> Orchestrator
    Orchestrator -->|1. Initialize| SG[Structure Generator]
    SG -->|Candidate Structures| Oracle

    subgraph Active Learning Loop
        Oracle -->|DFT Data (Energy/Forces)| Dataset[Dataset & Active Set]
        Dataset --> Trainer
        Trainer -->|Potential (YACE)| DE[Dynamics Engine]
        DE -->|Uncertainty Halt| Selection[Structure Selection]
        Selection -->|Extracted Clusters| Oracle
    end

    Trainer -->|Candidate Potential| Validator
    Validator -->|Pass/Fail| Orchestrator

    Orchestrator -->|Final Pot| Deployment
```

## Prerequisites

*   **Python**: 3.11 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **External Tools**:
    *   **LAMMPS**: For Molecular Dynamics simulations.
    *   **Quantum Espresso**: For DFT calculations.
    *   **Pacemaker**: For ACE potential training.

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-autopipec.git
    cd mlip-autopipec
    ```

2.  **Install dependencies**:
    Using `uv`:
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -e .[dev]
    ```

3.  **Environment Setup**:
    Copy the example environment file (if available) or ensure external tools are in your `PATH`.
    ```bash
    # Example: Export paths to external binaries
    export ASE_ESPRESSO_COMMAND="mpirun -np 4 pw.x -in PREFIX.pwi > PREFIX.pwo"
    export LAMMPS_COMMAND="lmp_serial"
    ```

## Usage

### Quick Start

1.  **Initialize a new project**:
    ```bash
    mlip-auto init
    ```
    This creates a `config.yaml` in your current directory.

2.  **Edit Configuration**:
    Open `config.yaml` and set your target composition (e.g., `Ti3Al`) and computational resources.

3.  **Run the Active Learning Loop**:
    ```bash
    mlip-auto run-loop
    ```
    The system will start the cycle: generating initial structures, running DFT, training the first generation potential, and iteratively improving it.

## Development Workflow

This project follows a strict development cycle.

*   **Run Tests**:
    ```bash
    pytest
    ```

*   **Linting & Type Checking**:
    The project enforces strict code quality using `ruff` and `mypy`.
    ```bash
    ruff check src tests
    mypy src
    ```

## Project Structure

```ascii
src/mlip_autopipec/
├── app.py                      # CLI Entry Point
├── domain_models/              # Pydantic Schemas (Config, Structure, Workflow)
├── orchestration/              # Core Logic (Workflow Manager, Phases)
├── modules/                    # Component Modules
│   ├── structure_gen/          # Structure Generation
│   ├── oracle/                 # DFT Interface (QE)
│   ├── trainer/                # Pacemaker Wrapper
│   └── validator/              # Validation Suite
└── inference/                  # Dynamics Engines (LAMMPS, EON)
dev_documents/                  # Documentation & Specifications
tests/                          # Test Suite
```

## License

This project is licensed under the MIT License.

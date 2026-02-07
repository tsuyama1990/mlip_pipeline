# PYACEMAKER: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Democratizing Atomic Simulations.** PYACEMAKER is a high-efficiency system that allows researchers to construct "State-of-the-Art" Machine Learning Interatomic Potentials (MLIP) with minimal effort. By automating the entire lifecycle—from Active Learning and DFT calculations to validation and deployment—it enables complex simulations like **Fe/Pt deposition on MgO** without requiring deep expertise in data science.

## 🚀 Key Features

*   **Zero-Config Workflow**: Define your material system in a single YAML file and let the autonomous agent handle the rest.
*   **Data Efficiency**: Uses Active Learning to select only the most informative structures, reducing expensive DFT calculations by >90%.
*   **Physics-Informed Robustness**: Hybrid potentials (ACE + ZBL/LJ) ensure simulations never crash due to non-physical atomic overlaps.
*   **Time-Scale Bridging**: Seamlessly integrates Molecular Dynamics (MD) and Kinetic Monte Carlo (kMC) to explore phenomena from picoseconds to seconds.
*   **Self-Healing Oracle**: Automatically recovers from DFT convergence failures without user intervention.

## 🏗️ Architecture Overview

The system follows a modular "Hub-and-Spoke" architecture orchestrated by a central Python controller.

```mermaid
flowchart TD
    Config[/Config.yaml/] --> Orchestrator

    subgraph Cycle [Active Learning Cycle]
        direction TB
        Orchestrator -->|1. Deploy Potential| Dynamics[Dynamics Engine\n(MD / kMC)]
        Dynamics -->|2. Halt & Extract| Generator[Structure Generator]
        Generator -->|3. Candidates| Oracle[Oracle\n(DFT / QE)]
        Oracle -->|4. Labeled Data| Trainer[Trainer\n(Pacemaker)]
        Trainer -->|5. New Potential| Validator[Validator]
        Validator -->|6. Pass/Fail| Orchestrator
    end

    Orchestrator -->|Final Output| Production[Production Potential]

    style Orchestrator fill:#f9f,stroke:#333,stroke-width:2px
    style Cycle fill:#e1f5fe,stroke:#333,stroke-dasharray: 5 5
```

## 📋 Prerequisites

*   **Python**: 3.12 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **External Tools** (Optional for Mock Mode, Required for Production):
    *   Quantum Espresso (`pw.x`)
    *   LAMMPS (`lmp_serial` / `lmp_mpi`) with USER-PACE package
    *   Pacemaker (`pace_train`, `pace_activeset`)
    *   EON (`eonclient`)

## 🛠️ Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies**:
    Using `uv` (fastest):
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -e .[dev]
    ```

3.  **Verify installation**:
    ```bash
    uv run pytest tests/
    ```

## ⚡ Usage

### Quick Start (Mock Mode)

To verify the pipeline logic without running heavy physics codes:

1.  **Initialise a project**:
    ```bash
    uv run pyacemaker init my_first_project
    cd my_first_project
    ```

2.  **Run the Active Learning Loop**:
    ```bash
    uv run pyacemaker run --config config.yaml --mode mock
    ```

3.  **Check Status**:
    ```bash
    uv run pyacemaker status
    ```

### Production Run

Edit `config.yaml` to set `type: qe` and `type: pacemaker`, then run:

```bash
uv run pyacemaker run --config config.yaml
```

## 💻 Development Workflow

This project follows the **AC-CDD** (Architecturally-Constrained Cycle-Driven Development) methodology.

*   **Linting**: strict type checking and style enforcement.
    ```bash
    uv run ruff check .
    uv run mypy .
    ```
*   **Testing**:
    ```bash
    uv run pytest --cov=src
    ```

### Project Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Data Models (Structure, Potential)
├── interfaces/             # Abstract Base Classes (Oracle, Trainer)
├── infrastructure/         # Concrete Implementations (QE, LAMMPS)
├── orchestrator/           # Core Logic
└── main.py                 # CLI Entry Point

dev_documents/              # AC-CDD Documentation
├── system_prompts/         # Cycle Specifications
└── ALL_SPEC.md             # Original Requirements
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

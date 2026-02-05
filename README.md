# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous system for constructing and operating State-of-the-Art Machine Learning Interatomic Potentials (MLIP) using the Pacemaker (ACE) framework. It lowers the barrier to entry for high-accuracy material simulations by automating the "Generation -> Calculation (DFT) -> Training" loop.

## Key Features

1.  **Zero-Config Workflow**: Define your material system in a single YAML file. The system handles the rest, from initial sampling to final validation.
2.  **Data Efficiency**: Uses Active Learning (D-Optimality) to select only the most informative structures, reducing DFT costs by >90% compared to random sampling.
3.  **Physics-Informed Robustness**: Automatically enforces core-repulsion using Delta Learning with ZBL/LJ baselines, preventing simulation crashes.
4.  **Scalability**: Seamlessly transitions from local exploration to massive-scale simulations on HPC, supporting MD and Adaptive Kinetic Monte Carlo (aKMC).
5.  **Self-Healing Oracle**: Automatically fixes common DFT convergence errors without user intervention.

## Architecture Overview

The system follows a Hub-and-Spoke architecture centered around an intelligent Orchestrator.

```mermaid
graph TD
    subgraph "Orchestration Layer"
        Orch[Orchestrator]
        Config[Global Config]
    end

    subgraph "Core Modules"
        SG[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        DE[Dynamics Engine (MD/kMC)]
        Val[Validator]
    end

    subgraph "Data Store"
        Pot[Potential (.yace)]
        Data[Dataset (.pckl)]
    end

    Config --> Orch
    Orch --> SG
    Orch --> DE

    SG -- "Candidate Structures" --> Oracle
    DE -- "High Uncertainty Structures" --> Oracle

    Oracle -- "Labeled Data (E, F, V)" --> Data
    Data --> Trainer
    Trainer -- "New Potential" --> Pot
    Pot --> Val
    Val -- "Pass/Fail" --> Orch
    Pot --> DE
```

## Prerequisites

*   **Python**: 3.12 or higher.
*   **Package Manager**: `uv` (recommended) or `pip`.
*   **External Engines**:
    *   **Quantum Espresso** (`pw.x`) for DFT calculations.
    *   **Pacemaker** (`pace_train`, `pace_activeset`) for training.
    *   **LAMMPS** (`lmp`) with `USER-PACE` package for MD.
    *   **EON** (optional) for aKMC.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    # Or with pip:
    # pip install -e ".[dev]"
    ```

3.  **Setup Configuration**:
    Copy the example configuration:
    ```bash
    cp config/example.yaml config.yaml
    ```

## Usage

### Quick Start (Mock Mode)
To verify the installation without running heavy calculations, run the mock loop:

```bash
uv run mlip-pipeline run config_mock.yaml
```

### Real Scientific Workflow
To run the "Fe/Pt on MgO" scenario:

1.  **Configure**: Edit `config.yaml` to specify `elements: ["Mg", "O", "Fe", "Pt"]`.
2.  **Run**:
    ```bash
    uv run mlip-pipeline run config.yaml
    ```
3.  **Monitor**: Check `active_learning/` directory for logs and intermediate results.

## Development

We follow a strictly defined development cycle using `uv` and `ruff`.

### Running Tests
```bash
uv run pytest
```

### Linting & Formatting
```bash
uv run ruff check .
uv run mypy .
```

### Project Structure
```ascii
src/mlip_autopipec/
├── config/             # Configuration schemas (Pydantic)
├── domain_models/      # Data structures (Atoms, Dataset)
├── interfaces/         # Core Protocols
├── orchestration/      # Main Loop & Mocks
├── structure_generation/
├── oracle/             # DFT & Embedding
├── training/           # Pacemaker Wrapper
└── dynamics/           # LAMMPS & EON
```

## License

This project is licensed under the MIT License.

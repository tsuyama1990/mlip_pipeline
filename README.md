# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an automated system for constructing robust Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) formalism. It provides a "Zero-Config" workflow that autonomously iterates through structure generation, property calculation, training, and validation.

## Key Features

-   **Automated Active Learning Loop**: Orchestrates the full cycle of structure generation, labeling, training, and validation.
-   **Mock Simulation Mode**: Includes a "Mock Mode" to verify workflow logic and configuration without requiring external physics engines (Cycle 01 verified).
-   **Strict Configuration Validation**: Ensures all input parameters are valid before execution using rigorous schema validation.
-   **Extensible Architecture**: Modular design allowing easy integration of various DFT codes (Quantum Espresso, VASP) and exploration strategies.

## Architecture Overview

The system is orchestrated by a central Python controller that manages a loop of specialized workers.

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]
    Orch -->|Manage| Loop{Active Learning Loop}

    subgraph "Core Modules"
        Explorer[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        Dyn[Dynamics Engine (LAMMPS/EON)]
        Val[Validator]
    end

    Loop -->|1. Request Structures| Explorer
    Explorer -->|Candidate Structures| Loop

    Loop -->|2. Request Data| Oracle
    Oracle -->|Labeled Data (E, F, S)| Loop

    Loop -->|3. Train| Trainer
    Trainer -->|Potential (.yace)| Loop

    Loop -->|4. Simulate & Check| Dyn
    Dyn -->|Uncertainty / Halt| Loop

    Loop -->|5. Verify| Val
    Val -->|Pass/Fail| Loop

    Loop -->|Final Output| Result[Production Potential]
```

## Prerequisites

-   **Python 3.12+**
-   **uv** (recommended for dependency management)

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Initialize environment**:
    ```bash
    uv sync
    ```

## Usage

To run the pipeline, use the CLI command:

```bash
uv run mlip-pipeline run config.yaml
```

### Quick Start Example

A sample configuration for a mock run:

```yaml
execution_mode: mock
max_cycles: 3
project_name: test_project
exploration:
  max_structures: 5
dft:
  calculator: espresso
training:
  potential_type: ace
```

Save this as `config.yaml` and run:

```bash
uv run mlip-pipeline run config.yaml
```

## Project Structure

```ascii
src/mlip_autopipec/
├── config/                  # Configuration Models
├── domain_models/           # Pydantic Data Classes
├── interfaces/              # Protocol Definitions
├── orchestration/           # Main Loop Logic
├── services/                # Business Logic
├── utils/                   # Shared Utilities
└── main.py                  # CLI Entrypoint

dev_documents/
├── system_prompts/          # Architectural Specs
└── tutorials/               # Jupyter Notebooks (UAT)
```

## License

This project is licensed under the MIT License.

# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.12%2B-blue)

**PYACEMAKER** (package: `mlip_autopipec`) is a "Zero-Config" active learning system designed to democratise the creation of state-of-the-art Machine Learning Interatomic Potentials (MLIPs). By orchestrating the interaction between Structure Generators, DFT Oracles (Quantum Espresso), and the Pacemaker training engine, it allows researchers to generate robust, physics-informed potentials with minimal human intervention.

## Key Features

*   **Autonomous Active Learning**: A closed-loop system that explores, labels (DFT), and trains (ACE) without manual hand-holding.
*   **Physics-Informed Robustness**: Enforces hard physical baselines (ZBL/LJ) and monitors real-time uncertainty to prevent simulation crashes.
*   **Adaptive Exploration**: Uses intelligent policies to sample high-value structures (high temperature, strain, defects) rather than random noise.
*   **Self-Healing Workflows**: Automatically recovers from DFT convergence failures and MD instabilities.
*   **Scalable Architecture**: Designed to run on everything from a laptop (Mock Mode) to an HPC cluster (Production Mode).

## Architecture Overview

PYACEMAKER follows a Hub-and-Spoke architecture, with a central Orchestrator managing specialised components.

```mermaid
graph TD
    User[User / Config] -->|Initialises| Orch[Orchestrator]

    subgraph "Core Components"
        Orch -->|Configures| SG[Structure Generator]
        Orch -->|Requests Data| Oracle[Oracle (DFT)]
        Orch -->|Manages| Trainer[Trainer (Pacemaker)]
        Orch -->|Deploys Potential| Dyn[Dynamics Engine]
        Orch -->|Validates| Val[Validator]
    end

    subgraph "Data Flow (Active Learning)"
        SG -->|Candidates| Oracle
        Dyn -->|Halt: High Uncertainty| Orch
        Orch -->|Diagnosis & Selection| SG
        Oracle -->|Labelled Data| Trainer
        Trainer -->|New Potential.yace| Dyn
        Trainer -->|New Potential.yace| Val
    end
```

## Prerequisites

*   **Python**: 3.12 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **External Engines** (Optional for Mock Mode):
    *   `pw.x` (Quantum Espresso) for DFT
    *   `pace_train` (Pacemaker) for training
    *   `lmp_serial` (LAMMPS) for MD

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-autopipec.git
    cd mlip-autopipec
    ```

2.  **Install dependencies**:
    Using `uv` (Recommended):
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -e .[dev]
    ```

3.  **Setup Environment**:
    Copy the example environment file (if available) or ensure your external tools are in your `$PATH`.

## Usage

### Quick Start (Mock Mode)

To verify the installation without running heavy calculations:

1.  **Initialise a new project**:
    ```bash
    mlip-auto init
    ```
    This creates a default `config.yaml`.

2.  **Run the Loop**:
    ```bash
    mlip-auto run-loop
    ```
    Watch the logs as the Orchestrator cycles through Exploration, Labelling, and Training using mock components.

### Production Run

Edit `config.yaml` to switch component types to `real`:

```yaml
orchestrator:
  work_dir: "experiments/fe_pt"
  max_iterations: 10

generator:
  type: adaptive
  policy:
    temperature_schedule: [[300, 1000]]

oracle:
  type: qe
  command: "mpirun -np 32 pw.x"

trainer:
  type: pace
  dataset_path: "data/training.pckl.gzip"
```

Then run `mlip-auto run-loop`.

## Development Workflow

We follow a strict development lifecycle divided into 8 cycles.

1.  **Run Tests**:
    ```bash
    pytest
    ```

2.  **Lint Code**:
    ```bash
    ruff check src tests
    mypy src
    ```

## Project Structure

```ascii
src/mlip_autopipec/
├── main.py                     # CLI Entry Point
├── config.py                   # Global Configuration
├── core/                       # Orchestrator & State Management
├── domain_models/              # Pydantic Schemas
└── components/                 # Functional Modules
    ├── generators/             # Structure Generation
    ├── oracle/                 # DFT Interface
    ├── training/               # Pacemaker Interface
    ├── dynamics/               # MD Engine
    └── validation/             # Physics Validation
dev_documents/
├── system_prompts/             # Cycle Specifications
└── FINAL_UAT.md                # User Acceptance Test Plan
```

## License

This project is licensed under the MIT License.

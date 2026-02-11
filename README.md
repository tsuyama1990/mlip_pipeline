# PyAceMaker

**Democratising Machine Learning Interatomic Potentials (MLIP) with Zero-Config Active Learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/mlip-pipeline/pyacemaker/ci.yml)](https://github.com/mlip-pipeline/pyacemaker/actions)

## 🚀 Overview

**PyAceMaker** is an automated pipeline designed to bridge the gap between complex atomic simulations and user-friendly potential generation. Built on top of the powerful **Pacemaker** (Atomic Cluster Expansion) engine, it enables researchers to create "State-of-the-Art" machine learning potentials with a single configuration file.

Traditionally, constructing an MLIP required deep expertise in DFT, MD, and fitting algorithms. PyAceMaker automates the entire loop: from adaptive structure generation and DFT labeling (Oracle) to Active Learning training and final validation.

## ✨ Features (Cycle 01)

*   **Core Orchestrator**: A robust state machine managing the active learning lifecycle.
*   **Type-Safe Configuration**: Strict validation of `config.yaml` using Pydantic V2 to prevent runtime errors.
*   **Mock Workflow**: Built-in mock components (Generator, Oracle, Trainer) to verify pipeline logic without heavy physics engines.
*   **CLI Tools**: Simple commands to initialize projects (`init`) and execute loops (`run-loop`).
*   **State Persistence**: Automatically saves and resumes workflow state (`workflow_state.json`), ensuring data safety.

## 🛠️ Requirements

*   **Python 3.12+**
*   **uv** (Recommended package manager)
*   **Dependencies**: `ase`, `numpy`, `pydantic`, `typer`, `pyyaml`.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mlip-pipeline/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Verify installation**:
    ```bash
    uv run pyacemaker --help
    ```

## 🚦 Usage

### 1. Initialize a Project
Create a new project directory with a default configuration.

```bash
uv run pyacemaker init my_project
cd my_project
```

This creates:
*   `config.yaml`: Default configuration file.
*   `data/`: Directory for storing structures and potentials.

### 2. Run the Mock Loop
Execute the active learning loop using the default mock configuration.

```bash
uv run pyacemaker run-loop
```

You should see logs indicating the progression of cycles:
```
INFO - Cycle 1 Started
INFO - Step 1: Structure Generation
INFO - Step 2: Labeling (Oracle)
INFO - Step 3: Training
INFO - Step 4: Validation
INFO - Cycle 1 Completed
```

### 3. Customize Configuration
Edit `config.yaml` to change parameters (e.g., number of iterations, mock noise).

```yaml
orchestrator:
  work_dir: "/absolute/path/to/my_project"
  n_iterations: 10

generator:
  type: mock
  n_candidates: 50
```

## 🏗️ Architecture

PyAceMaker follows a modular "Hub-and-Spoke" architecture:

```mermaid
graph TD
    User[User] -->|Config (yaml)| Orch[Orchestrator]
    Orch -->|Manage| State[State Manager]

    subgraph "Active Learning Loop"
        Orch -->|1. Explore| Gen[Structure Generator]
        Orch -->|2. Simulate & Halt| Dyn[Dynamics Engine]
        Orch -->|3. Label| Oracle[Oracle (DFT)]
        Orch -->|4. Train| Trainer[Trainer (Pacemaker)]
        Orch -->|5. Verify| Valid[Validator]
    end
```

## 📂 Project Structure

```
.
├── src/
│   └── mlip_autopipec/     # Source code
│       ├── components/     # Logic modules (Mock implementations available)
│       ├── core/           # Orchestrator, Logger, State Manager
│       └── domain_models/  # Pydantic Schemas (Config, Inputs, Results)
├── tests/                  # Pytest suite (Unit & E2E)
├── pyproject.toml          # Config & Dependencies
└── README.md
```

## 📄 License

This project is licensed under the MIT License.

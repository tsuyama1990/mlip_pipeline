# MLIP Auto PiPEC: Automated Machine Learning Interatomic Potential Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**MLIP Auto PiPEC** is a fully automated, active-learning based system for generating state-of-the-art Machine Learning Interatomic Potentials (MLIPs). It democratizes access to high-accuracy atomic simulations by replacing manual, expert-driven workflows with a robust, "Zero-Config" autonomous pipeline.

## Features

*   **Robust Configuration**: Strictly validated YAML configuration using Pydantic schemas ensures fail-fast behavior.
*   **Database Management**: Thread-safe, resilient interface to `ase.db` (SQLite) for storing structures and calculation results.
*   **Self-Healing**: Automated recovery strategies for DFT convergence failures.
*   **Active Learning Loop**: Autonomous cycle of generation, labeling, training, and validation.

## Requirements

*   **Python**: 3.11+
*   **Dependencies**: `ase`, `numpy`, `pydantic`, `typer`, `rich`, `pyyaml`.
*   **External Engines**: Quantum Espresso, LAMMPS, Pacemaker.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/mlip-autopipec.git
    cd mlip-autopipec
    ```

2.  **Install Dependencies**
    Using `uv` (Recommended):
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install .
    ```

## Usage

### 1. Initialize Project
Create a template configuration file (`input.yaml`) with default settings:
```bash
mlip-auto init
```

### 2. Configure System
Edit `input.yaml` to define your target system (e.g., Fe-Ni alloy) and computational parameters:
```yaml
target_system:
  elements: ["Fe", "Ni"]
  composition: {"Fe": 0.7, "Ni": 0.3}
dft:
  pseudopotential_dir: "/path/to/upf"
  ecutwfc: 40.0
```

### 3. Validate Configuration
Ensure your configuration is valid before running expensive calculations:
```bash
mlip-auto validate input.yaml
```

### 4. Initialize Database (Optional)
Initialize the SQLite database (`mlip.db`):
```bash
mlip-auto db init --config input.yaml
```

## Architecture

The project is structured as follows:

```ascii
src/mlip_autopipec/
├── app.py                      # CLI Entry Point
├── config/                     # Configuration Schemas (Pydantic)
├── data_models/                # Core Data Structures (Atoms, Candidates)
├── orchestration/              # Database & Workflow Management
├── utils/                      # Logging & Utilities
└── ...                         # Feature Modules (DFT, Training, Generator)
```

## Roadmap

- [x] **Cycle 01**: Core Framework, Config, Database.
- [ ] **Cycle 02**: Structure Generation.
- [ ] **Cycle 03**: DFT Oracle Interface.
- [ ] **Cycle 04**: Training Orchestration.
- [ ] **Cycle 05**: Inference & Active Learning.

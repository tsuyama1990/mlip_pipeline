# PYACEMAKER

**Automated Machine Learning Interatomic Potential Construction System**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

**PYACEMAKER** is a next-generation pipeline designed to democratize the creation of high-accuracy Machine Learning Interatomic Potentials (MLIPs). It solves the "data efficiency" problem by leveraging **MACE Knowledge Distillation**. Instead of running thousands of expensive DFT calculations, PYACEMAKER uses a pre-trained Large Foundation Model (MACE-MP) as a "Teacher" to guide the exploration of chemical space and train a fast, production-ready "Student" potential (ACE/Pacemaker).

Ideally suited for materials scientists who need DFT-level accuracy with the speed of classical molecular dynamics.

## Key Features (Cycle 02 Verified)

-   **Intelligent Exploration**: DIRECT sampling (MaxMin diversity) for initial structure generation to maximize coverage.
-   **Active Learning**: Uncertainty-based selection of structures using MACE ensemble variance or heuristics.
-   **MACE Integration**: Seamlessly load and use MACE-MP-0 foundation models as surrogate oracles.
-   **Configurable Pipeline**: Robust YAML-based configuration with schema validation (Pydantic).
-   **Mock Mode**: Fully functional mock execution for CI/CD and testing without expensive hardware.
-   **Modular Architecture**: Extensible design separating Oracle, Trainer, and Generator components.
-   **Automated Validation**: Built-in integrity checks for structures and datasets.

## Architecture Overview

The system operates as a centralized Orchestrator managing specialized modules for Generation, Oracle evaluation (MACE/DFT), and Training.

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]

    subgraph "Phase 1: Active Learning"
        Orch -->|Step 1: DIRECT| Gen[Structure Generator]
        Gen -->|Structures| AL[Active Learning Loop]
        AL -->|Step 2: Uncertainty| MACE_O[MACE Oracle]
        MACE_O -->|High Variance| DFT[DFT Oracle]
        DFT -->|Truth Labels| DB[(DFT Dataset)]
    end

    subgraph "Phase 2: Distillation"
        Orch -->|Step 3: Fine-tune| MACE_T[MACE Trainer]
        DB --> MACE_T
        MACE_T -->|Fine-tuned Model| MACE_MD[MACE Dynamics]
        Orch -->|Step 4: Surrogate Sampling| MACE_MD
        MACE_MD -->|Structures| Label[Surrogate Labeler]
        MACE_T -->|Predict| Label
        Label -->|Pseudo Labels| S_DB[(Surrogate Dataset)]
    end

    subgraph "Phase 3: ACE Training"
        Orch -->|Step 6: Base Train| PACE_T[Pacemaker Trainer]
        S_DB --> PACE_T
        PACE_T -->|Base Potential| PACE_Model

        Orch -->|Step 7: Delta Learning| Delta[Delta Learner]
        DB -->|Real Labels| Delta
        PACE_Model --> Delta
        Delta -->|Final Potential| Final_ACE[(Final ACE.yace)]
    end
```

## Prerequisites

-   **Python 3.11+**
-   **uv** (Recommended for dependency management) or pip.
-   **MACE-MP-0** (automatically downloaded or provided via path).
-   **Pacemaker** (external binary or library).
-   **Optional**: VASP or Quantum Espresso for "Real Mode" execution.

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies**:
    We use `uv` for fast dependency resolution.
    ```bash
    uv sync
    ```
    Or using pip:
    ```bash
    pip install .
    ```

## Usage

### Quick Start

1.  **Create a configuration file** (`config.yaml`):
    ```yaml
    version: "0.1.0"
    project:
      name: "MyFirstPotential"
      root_dir: "/abs/path/to/project"
    oracle:
      # Use MACE as surrogate
      mace:
        model_path: "medium" # or path to local .model file
        device: "cuda"       # or "cpu"
      # DFT Configuration (Optional if running pure MACE or Mock)
      dft:
        code: "quantum_espresso"
        pseudopotentials:
          Fe: "Fe.pbe.UPF"
    ```

2.  **Run the pipeline**:
    ```bash
    uv run pyacemaker run config.yaml
    ```

3.  **Monitor Progress**:
    The system will log its progress. Check the `data/` directory for artifacts like `dataset.pckl.gzip`.

### Quick Validation

To verify the installation and see the system in action (Mock Mode), run the UAT script:

```bash
uv run pytest tests/uat/test_cycle02_features.py
```

This ensures that the configuration loading, MACE integration (mock), and orchestrator workflow are functioning correctly.

## Configuration Reference

A complete `config.yaml` example:

```yaml
version: "0.1.0"

project:
  name: "Fe_C_System"
  root_dir: "/home/user/projects/fe_c"

logging:
  level: "INFO"

# Oracle Settings
oracle:
  mock: false  # Set to true for testing without GPU/DFT
  mace:
    model_path: "medium"  # Downloads MACE-MP-0 medium
    device: "cuda"
    default_dtype: "float64"
    batch_size: 32
  dft:
    code: "quantum_espresso"
    command: "mpirun -np 4 pw.x"
    pseudopotentials:
      Fe: "/path/to/pseudos/Fe.pbe.UPF"
      C: "/path/to/pseudos/C.pbe.UPF"
    kspacing: 0.04
    smearing: 0.02

# Generator Settings
structure_generator:
  strategy: "adaptive" # or "random"

# Orchestrator Settings
orchestrator:
  max_cycles: 10
  uncertainty_threshold: 0.1
  n_local_candidates: 20
```

## Troubleshooting

### Common Issues

1.  **`ConfigurationError: MaceManager is None but mock is False`**
    *   **Cause**: You have enabled `mace` in config but `mock: false` and the `mace` library is not installed or failed to load.
    *   **Fix**: Ensure `mace-torch` is installed (`pip install mace-torch`) or set `oracle.mock: true`.

2.  **`ValueError: Checksum verification failed`**
    *   **Cause**: Data corruption or file modification during write.
    *   **Fix**: Ensure you have write permissions to the data directory. The system automatically handles checksums now; try deleting the corrupt `.pckl.gzip` and `.sha256` files to restart.

3.  **`FileNotFoundError: Dataset file not found`**
    *   **Cause**: Cold start failed or path is incorrect.
    *   **Fix**: Ensure `project.root_dir` is absolute and exists. The Orchestrator will attempt to generate initial structures if the dataset is missing.

## Development Workflow

This project follows a strict cycle-based development process.

### Running Tests
```bash
uv run pytest
```

### Linting & Formatting
We enforce strict code quality using `ruff` and `mypy`.
```bash
uv run ruff check .
uv run mypy .
```

## Project Structure

```text
pyacemaker/
├── dev_documents/          # Specs and Architecture docs
├── src/
│   └── pyacemaker/         # Source code
│       ├── core/           # Base classes & Config
│       ├── oracle/         # MACE & DFT interfaces
│       ├── modules/        # Module implementations (Oracle, Trainer, etc.)
│       ├── trainer/        # Pacemaker & MACE training
│       ├── generator/      # Structure generation
│       ├── main.py         # CLI Entry Point
│       └── orchestrator.py # Main logic
├── tests/                  # Unit and Integration tests
├── tutorials/              # Executable tutorials
├── pyproject.toml          # Project configuration
└── README.md
```

## License

MIT License. See `LICENSE` for details.

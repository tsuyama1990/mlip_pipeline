# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous system for constructing and operating State-of-the-Art Machine Learning Interatomic Potentials (MLIP) using the Pacemaker (ACE) framework. It lowers the barrier to entry for high-accuracy material simulations by automating the "Generation -> Calculation (DFT) -> Training" loop.

## Key Features

*   **Automated Orchestration**: Hub-and-Spoke architecture managing the data flow between Structure Generation, Oracle, and Trainer.
*   **Mock Loop Verification**: Includes a simulation mode to verify the logic flow without expensive computations.
*   **Type-Safe Design**: Built with strictly typed Python 3.12+ (Pydantic, Typer) for robustness.
*   **Active Learning Ready**: Designed to support iterative improvement of potentials (Cycle 1 implementation focuses on the core framework).

## Prerequisites

*   **Python**: 3.12 or higher.
*   **Package Manager**: `uv` (recommended).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd mlip-pipeline
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

## Usage

### Quick Start (Core Framework Verification)

1.  **Create a Configuration File**:
    Create a file named `config.yaml` with the following content:
    ```yaml
    work_dir: "./workspace"
    max_cycles: 3
    random_seed: 42
    ```

2.  **Run the Pipeline**:
    Execute the command to start the orchestration loop. Currently, this runs with Mock components to verify the system architecture.
    ```bash
    uv run mlip-pipeline run --config config.yaml
    ```

3.  **Output**:
    You should see logs indicating the progress of the active learning cycles (Generation -> Calculation -> Training -> Validation).

## Architecture

```ascii
src/mlip_autopipec/
├── config/             # Configuration schemas (GlobalConfig)
├── domain_models/      # Data structures (StructureMetadata, Dataset)
├── interfaces/         # Core Protocols (Explorer, Oracle, Trainer, Validator)
├── orchestration/      # Main Orchestrator Logic & Mock Implementations
├── utils/              # Utilities (Logging)
└── main.py             # CLI Entry Point
```

## Development

### Running Tests
```bash
uv run pytest
```

### Linting & Formatting
```bash
uv run ruff check .
uv run mypy .
```

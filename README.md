# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: Core Framework & User Interface

This cycle establishes the robust infrastructure for the MLIP-AutoPipe ecosystem. It implements the configuration management system, data persistence layer, and centralized logging.

### Key Features:
- **Strict Configuration Schema**: Implements a "Schema-First" approach using Pydantic V2. The `UserInputConfig` and `SystemConfig` models ensure that all inputs are validated before execution.
- **Project Initialization**: A new `init` command (`mlip-auto init`) sets up the project directory structure, initializes the database, and configures logging.
- **Database Interface**: A `DatabaseManager` wrapper around `ase.db` enforces schema compliance and metadata tracking for provenance.
- **Centralized Logging**: A unified logging system using `rich` for console output and file-based logging for archival.

## Cycle 02: Automated DFT Factory (Planned)

This cycle will implement the cornerstone of the MLIP-AutoPipe system: a robust and autonomous DFT calculation factory. This module is responsible for taking an atomic structure and reliably returning its DFT-calculated properties, handling the complexities of the underlying quantum mechanics engine.

## Cycle 03: The Training Engine (Planned)

This cycle implements the `PacemakerTrainer`, the component responsible for consuming the DFT data and producing a trained Machine Learning Interatomic Potential (MLIP).

## Cycle 04: Active Learning with On-The-Fly Inference (Planned)

This cycle introduces the `LammpsRunner`, a key component for active learning. This module runs molecular dynamics (MD) simulations using the trained MLIP and detects when the model's predictions are uncertain, signaling the need for new DFT data.

## Cycle 05: Configuration-Driven Workflow (Planned)

This cycle introduces a user-friendly, configuration-driven workflow.

## Cycle 06: Resilience and Scalability (Planned)

This cycle transforms the workflow from a linear process into a robust, high-throughput data generation factory.

## Cycle 07: User Interface (CLI) (Planned)

This cycle focuses on usability, providing a polished and professional Command Line Interface (CLI) for the application.

## Cycle 08: Monitoring and Usability (Planned)

This cycle adds critical introspection capabilities to the workflow, allowing users to monitor the progress and performance of their long-running active learning jobs.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

Initialize a new project:

```bash
mlip-auto init input.yaml
```

Run the workflow (Cycle 8):

```bash
mlip-auto run input.yaml
```

Check status:

```bash
mlip-auto status .
```

# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: Core Framework & User Interface

This cycle lays the foundation for the entire MLIP-AutoPipe ecosystem. It establishes the configuration management system, the database interface, and the command-line interface (CLI).

### Key Features:
- **Strict Configuration Schema**: Uses Pydantic V2 to enforce rigorous validation of user inputs. The `MinimalConfig` (user-facing) and `SystemConfig` (internal) models ensure type safety and data integrity from the start.
- **Project Initialization**: The `mlip-auto run input.yaml` command initializes a new project workspace, creating the necessary directory structure, logging configuration, and database file.
- **Database & Provenance**: A `DatabaseManager` wraps `ase.db` (SQLite) to ensure that every project starts with a traceable configuration stored in the database metadata.
- **Centralized Logging**: A unified logging system using `rich` provides structured, readable output for both console and file logs.

## Cycle 02: Automated DFT Factory

This cycle implements the cornerstone of the MLIP-AutoPipe system: a robust and autonomous DFT calculation factory. This module is responsible for taking an atomic structure and reliably returning its DFT-calculated properties, handling the complexities of the underlying quantum mechanics engine.

### Key Features:
- **Schema-Driven Design**: All data structures for DFT jobs and results are rigorously defined using Pydantic, ensuring type safety and data integrity throughout the workflow.
- **Automated DFT Factory**: The `DFTFactory` class provides a high-level interface for running Quantum Espresso calculations. It encapsulates:
  - **Heuristic Parameter Generation**: Automatically determines optimal DFT parameters (e.g., cutoffs, k-points) based on the input structure's elements and geometry.
  - **Resilient Execution**: Includes an auto-recovery mechanism that can handle common DFT convergence failures by intelligently adjusting parameters and retrying the calculation.
- **Data Persistence**: All successful DFT results are saved to an ASE-compatible database, creating a structured and queryable training set for future machine learning cycles.

## Cycle 03: The Training Engine

This cycle implements the `PacemakerTrainer`, the component responsible for consuming the DFT data and producing a trained Machine Learning Interatomic Potential (MLIP). This bridges the gap between raw quantum mechanical results and a fast, accurate, and usable surrogate model.

### Key Features:
- **Automated Training**: The `PacemakerTrainer` class orchestrates the entire training workflow, including:
  - **Data Ingestion**: Automatically reads training data from the central ASE database.
  - **Dynamic Configuration**: Generates the necessary input files for the Pacemaker training code using a Jinja2 template.
  - **Secure Execution**: Invokes the Pacemaker training process in a secure, monitored subprocess.
- **Schema-Driven Configuration**: The entire training process is configured via a strict Pydantic model, ensuring that all hyperparameters are valid before the training process is launched.

## Cycle 04: Active Learning with On-The-Fly Inference

This cycle introduces the `LammpsRunner`, a key component for active learning. This module runs molecular dynamics (MD) simulations using the trained MLIP and detects when the model's predictions are uncertain, signaling the need for new DFT data.

### Key Features:
- **On-The-Fly Uncertainty Detection**: The `LammpsRunner` executes LAMMPS simulations and monitors the MLIP's uncertainty on-the-fly.
- **Data Extraction for DFT**: When high uncertainty is detected, the runner extracts the specific atomic configuration and its local environment for subsequent high-fidelity DFT calculations.
- **Periodic Embedding and Force Masking**: Implements sophisticated data extraction techniques to ensure that the data sent to the DFT factory is physically meaningful and periodic.

## Cycle 05: Configuration-Driven Workflow

This cycle introduces a user-friendly, configuration-driven workflow. The user now interacts with the system through a simple `input.yaml` file, and the `WorkflowManager` orchestrates the entire process based on this high-level input.

### Key Features:
- **User-Friendly Configuration**: A simple `input.yaml` file is now the single point of entry for the user, abstracting away the low-level details of the workflow.
- **Centralized Orchestration**: The `WorkflowManager` class takes the user's input and expands it into a comprehensive, validated `SystemConfig` object, which is then used to configure and run the entire active learning pipeline.
- **Rich Metadata**: The data persistence layer has been enhanced to store rich metadata, including unique IDs and the configuration type, for every DFT calculation, ensuring data provenance and traceability.

## Cycle 06: Resilience and Scalability

This cycle transforms the workflow from a linear process into a robust, high-throughput data generation factory. It introduces two critical features for real-world, long-duration simulations: resilience through checkpointing and scalability through parallel execution.

### Key Features:
- **Checkpointing and Recovery**: The `WorkflowManager` now periodically saves its entire state to a `checkpoint.json` file. If the workflow is interrupted for any reason (e.g., a system crash), it can be restarted and will seamlessly resume from the last saved state, preventing any loss of work.
- **Parallel Execution with Dask**: The system is now integrated with the Dask distributed computing library. The `WorkflowManager` acts as a dispatcher, submitting hundreds of independent DFT calculations to a Dask cluster for parallel execution. This dramatically reduces the time required for data generation.
- **Robust Retry Decorator**: The DFT auto-recovery logic has been refactored into a generic and extensible `@retry` decorator. It now supports domain-specific callbacks, allowing for intelligent, context-aware error recovery (such as modifying DFT parameters on convergence failure) while keeping the core retry logic reusable.

## Cycle 07: User Interface (CLI)

This cycle focuses on usability, providing a polished and professional Command Line Interface (CLI) for the application.

### Key Features:
- **`mlip-auto` CLI**: A unified entry point built with `Typer` and `rich`, allowing users to start workflows with a simple command: `mlip-auto run input.yaml`.
- **Helpful Feedback**: The CLI provides clear, color-coded output for success, errors, and progress updates.
- **Robust Validation**: Invalid configurations or command-line usage errors are caught immediately with helpful messages, ensuring a "fail-fast" experience before expensive computations begin.
- **Schema Integration**: The CLI fully integrates with the project's rigorous Pydantic schemas, ensuring that user input is validated against the same rules used by the backend.

## Cycle 08: Monitoring and Usability

This cycle adds critical introspection capabilities to the workflow, allowing users to monitor the progress and performance of their long-running active learning jobs.

### Key Features:
- **Status Dashboard**: A new `mlip-auto status` command generates a comprehensive HTML dashboard. This report provides:
  - **Key Metrics**: Real-time stats on the current generation, completed calculations, and pending jobs.
  - **Performance Tracking**: A plot of the model's Force RMSE over training generations, providing quantitative proof of learning.
  - **Dataset Insights**: A visualization of the dataset composition, showing how active learning is enriching the training data.
- **On-Demand Generation**: The dashboard is generated on-demand from the project's checkpoint and database, ensuring the user always gets the latest status without needing a persistent web server.
- **Enhanced Checkpointing**: The system state now tracks the full history of training metrics, enabling historical performance analysis.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

Initialize a new project:

```bash
mlip-auto run input.yaml
```

This will create a project directory, initialize the database, and set up logging.

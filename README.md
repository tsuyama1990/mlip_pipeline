# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: The Foundation - Automated DFT Factory

This cycle implements the cornerstone of the MLIP-AutoPipe system: a robust and autonomous DFT calculation factory. This module is responsible for taking an atomic structure and reliably returning its DFT-calculated properties, handling the complexities of the underlying quantum mechanics engine.

### Key Features:
- **Schema-Driven Design**: All data structures for DFT jobs and results are rigorously defined using Pydantic, ensuring type safety and data integrity throughout the workflow.
- **Automated DFT Factory**: The `DFTFactory` class provides a high-level interface for running Quantum Espresso calculations. It encapsulates:
  - **Heuristic Parameter Generation**: Automatically determines optimal DFT parameters (e.g., cutoffs, k-points) based on the input structure's elements and geometry.
  - **Resilient Execution**: Includes an auto-recovery mechanism that can handle common DFT convergence failures by intelligently adjusting parameters and retrying the calculation.
- **Data Persistence**: All successful DFT results are saved to an ASE-compatible database, creating a structured and queryable training set for future machine learning cycles.

## Cycle 02: Surrogate-First Exploration

This cycle introduces the crucial **`SurrogateExplorer`** module, which adds a layer of intelligence and efficiency to the data generation pipeline. It prevents the waste of expensive DFT resources on unphysical or redundant structures.

### Key Features:
- **Surrogate Pre-screening**: Uses a fast, pre-trained MACE model to perform a "sanity check" on candidate structures, immediately discarding any with unphysically high forces.
- **Intelligent Down-selection**: For the structures that pass screening, it employs Farthest Point Sampling (FPS) on structural fingerprints (SOAP) to select a small, maximally diverse subset for DFT calculation. This ensures that computational effort is focused on the most informative new structures.

## Cycle 03: The Training Engine

This cycle implements the `PacemakerTrainer`, the component responsible for consuming the DFT data and producing a trained Machine Learning Interatomic Potential (MLIP). This bridges the gap between raw quantum mechanical results and a fast, accurate, and usable surrogate model.

### Key Features:
- **Automated Training**: The `PacemakerTrainer` class orchestrates the entire training workflow, including:
  - **Data Ingestion**: Automatically reads training data from the central ASE database.
  - **Dynamic Configuration**: Generates the necessary input files for the Pacemaker training code using a Jinja2 template.
  - **Secure Execution**: Invokes the Pacemaker training process in a secure, monitored subprocess.
- **Schema-Driven Configuration**: The entire training process is configured via a strict Pydantic model, a
- ensuring that all hyperparameters are valid before the training process is launched.

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

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

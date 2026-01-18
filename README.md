# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: Core Framework & User Interface

This cycle lays the bedrock for the entire MLIP-AutoPipe ecosystem. Before any physics can be simulated or any potentials trained, we must establish a robust, type-safe, and verifiable infrastructure. The primary objective of this cycle is to implement the **Configuration Management System** and the **Core Utilities** (Database and Logging) that will serve as the nervous system of the application.

### Key Features:
- **Strict Schema**: A rigorous Pydantic V2 schema defines the `MinimalConfig` (user input) and `SystemConfig` (internal state), ensuring type safety and validity before any computation begins.
- **Data Persistence**: A `DatabaseManager` wrapper around `ase.db` ensures that no data is saved without its provenance metadata, enforcing traceability.
- **Centralized Logging**: A unified logging system provides structured, machine-parsable logs, essential for debugging autonomous workflows.
- **CLI Initialization**: The `mlip-auto run input.yaml` command initializes a project workspace, validates the configuration, and establishes the database connection.

## Cycle 02: Automated DFT Factory

This cycle implements the **Automated DFT Factory**, the "engine room" of the pipeline. It autonomously manages Quantum Espresso calculations, ensuring robust data generation for MLIP training.

### Key Features:
- **Autonomous Execution**: The `QERunner` manages the entire lifecycle of a DFT calculation, from input generation to output parsing.
- **Auto-Recovery**: A state-machine-based `RecoveryHandler` automatically detects and fixes common DFT errors (convergence failure, diagonalization errors) by adjusting parameters like mixing beta and temperature.
- **Physics-Informed Input Generation**: The `InputGenerator` automatically selects appropriate pseudopotentials (SSSP) and K-point grids based on cell density and material composition (e.g., handling magnetism for Fe, Co, Ni).
- **Data Standardization**: All results are encapsulated in a strictly typed `DFTResult` object, ensuring consistency across the pipeline.

## Cycle 03: Structure Generator

This cycle implements the **Structure Generator**, a system for creating diverse atomic structures for training data. It includes tools for generating SQS (Special Quasirandom Structures), applying random distortions (strain, rattle), creating defects (vacancies, interstitials), and generating Normal Mode Sampling (NMS) structures for molecules.

## Cycle 04: Surrogate Explorer

This cycle implements the **Surrogate Explorer**, a mechanism to intelligently select the most valuable structures for DFT calculation from a large pool of candidates.

### Key Features:
- **MACE Foundation Model**: Uses a pre-trained MACE-MP model as a "scout" to predict forces and filter out physically catastrophic structures (e.g., exploding atoms).
- **Diversity Sampling (FPS)**: Implements Farthest Point Sampling (FPS) using SOAP descriptors to select a geometrically diverse subset of structures, maximizing information gain for MLIP training.
- **Descriptor Calculation**: robust calculation of SOAP and ACE descriptors for both bulk and molecular systems.
- **Efficient Pipeline**: A `SurrogatePipeline` orchestrates the flow: Generation -> Pre-screening (MACE) -> Featurization (SOAP) -> Selection (FPS).

## Cycle 05: Active Learning & Training

This cycle implements the **Active Learning & Training** module, automating the creation of Machine Learning Potentials using Pacemaker.

### Key Features:
- **Dataset Preparation**: Automated export of training data from ASE-db, ensuring proper formatting and disjoint train/test splitting.
- **Delta Learning**: Implementation of ZBL (Ziegler-Biersack-Littmark) baseline subtraction to enforce correct short-range repulsion physics.
- **Pacemaker Integration**: A `PacemakerWrapper` that autonomously manages the training lifecycle, including dynamic configuration generation via Jinja2 templates and output monitoring.
- **Force Masking**: Support for masking forces on specific atoms (e.g., buffer regions) during training to prevent learning artifacts.

## Cycle 06: Scalable Inference Engine (Part 1)

This cycle implements the first part of the **Scalable Inference Engine**, enabling the system to run Molecular Dynamics simulations for "stress-testing" the trained potential.

### Key Features:
- **LAMMPS Wrapper**: A robust `LammpsRunner` that orchestrates MD simulations, handling input generation (script & data) and execution.
- **Active Learning Mining**: Logic to monitor the "Extrapolation Grade" ($\gamma$) in real-time. If the simulation wanders into uncertain regions ($\gamma > 5.0$), it automatically dumps the problematic configurations.
- **Uncertainty Quantification (UQ)**: An `UncertaintyChecker` module that parses MD dump files to extract and flag high-uncertainty structures for re-training.
- **Thermodynamic Analysis**: Basic analysis utilities to extract properties (Temperature, Pressure) from LAMMPS logs.

## Development Status

Cycle 01 through Cycle 06 features are implemented.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv sync --extra dev
```

Run the initialization:

```bash
mlip-auto run input.yaml
```

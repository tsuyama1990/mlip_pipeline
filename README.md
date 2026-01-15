# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: The Foundation

This first cycle establishes the architectural backbone of the project. It includes:

- **Pydantic Schemas**: A robust, schema-driven framework for managing all system configurations.
- **DFT Factory Core**: The initial implementation of the `QEProcessRunner`, a component for interacting with the Quantum Espresso DFT engine.
- **Database Manager**: A data persistence layer for handling ASE-compatible databases with custom metadata.

## Cycle 02: The Seed

This cycle introduces the `PhysicsInformedGenerator`, a module designed to solve the "cold-start" problem by creating an initial, diverse set of atomic structures.

- **Physics-Informed Generator**: A factory class that generates structures based on physical principles.
  - For alloys, it produces strained and thermally "rattled" structures.
  - For crystals, it creates structures with common point defects.
- **Mock Implementation**: Due to a dependency conflict with `icet` and Python 3.12, the generator is currently implemented as a mock. It returns hard-coded structures but preserves the full workflow, allowing for seamless integration with downstream modules.

## Cycle 03: The Filter

This cycle implements the crucial **`SurrogateExplorer`** module, which introduces the first layer of intelligence and cost-saving to the pipeline. It addresses the challenge of efficiently processing the large number of candidate structures generated in the previous cycle.

- **Two-Stage Filtering**: The explorer implements a sophisticated two-stage process to select the most valuable structures for expensive DFT calculations.
  - **Surrogate Screening**: It first uses a fast, pre-trained surrogate model (MACE) to quickly discard any structures that are unphysical (e.g., have excessively high energy).
  - **Intelligent Selection**: From the remaining candidates, it calculates structural descriptors (SOAP) and uses Farthest Point Sampling (FPS) to select a small, maximally diverse subset. This ensures that the DFT resources are spent on generating the most informative data possible.

## Cycle 04: The Factory

This cycle introduces the **`PacemakerTrainer`**, the module that closes the first major loop of the project: transforming raw data into a functional Machine Learning Interatomic Potential.

- **Automated Training**: The trainer automates the entire MLIP creation process.
  - It queries the central database to gather all DFT-calculated structures.
  - It dynamically generates the necessary configuration files for the Pacemaker training engine based on the project's `SystemConfig`.
  - It executes the training process, monitors for success, and manages the resulting potential file (`.yace`).
- **End-to-End Workflow**: With this cycle, the system can now perform an end-to-end workflow, from initial structure generation (Cycle 02), through intelligent data selection (Cycle 03), to the creation of a version-one MLIP. This provides the foundation for the full active learning loop to be implemented in subsequent cycles.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

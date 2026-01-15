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

## Cycle 05: The Intelligence

This cycle closes the loop, transforming the pipeline into a true active learning engine. It introduces the **`LammpsRunner`** and a main **`app.py`** orchestrator, enabling the system to autonomously improve its own MLIP.

- **Active Learning Loop**: The system can now run molecular dynamics simulations and intelligently detect when the simulation enters a region of high uncertainty.
- **Automated DFT Trigger**: Upon detecting high uncertainty, the system automatically pauses the simulation, takes the uncertain atomic structure, and sends it to the DFT queue for calculation.
- **Self-Improving Potential**: The new DFT data is then used to retrain the MLIP, creating a more robust potential. The simulation is then seamlessly resumed with the improved model. This "Zero-Human" protocol allows the system to explore conformational space and learn on the fly.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

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

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

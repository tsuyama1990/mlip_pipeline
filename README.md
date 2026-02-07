# PYACEMAKER (MLIP Pipeline)

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

**Automated Active Learning Pipeline for Machine Learning Interatomic Potentials.**

## Overview

### What is this?
PYACEMAKER is a modular, scalable pipeline for generating Machine Learning Interatomic Potentials (MLIPs). It automates the active learning loop: exploring atomic structures, calculating ground truth data (DFT), fitting potentials, and validating performance.

### Why?
Developing MLIPs requires complex workflows involving DFT calculations, MD simulations, and iterative training. This tool orchestrates these components into a seamless, fault-tolerant pipeline, allowing researchers to focus on physics rather than scripting.

## Features

- **Modular Architecture**: Swappable components for Oracle, Trainer, Dynamics, Generator, Validator, and Selector.
- **DFT Integration**: Built-in support for Quantum Espresso (QE) via ASE, with automatic self-healing (retries with adjusted parameters) for convergence failures.
- **Active Learning Loop**: Automated generation of candidate structures (e.g., via Random Displacement) and selection for labeling.
- **Scalability**: Designed for streaming data processing to handle large datasets without memory bottlenecks.
- **Robust CLI**: Easy-to-use command line interface for initialization, execution, and debugging (`compute` command).
- **Configuration Validation**: Strict schema validation using Pydantic ensures reliable runs.

## Requirements

- Python >= 3.12
- `uv` (recommended for dependency management) or `pip`
- Quantum Espresso (`pw.x`) installed and accessible in PATH (for DFT calculations)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mlip-pipeline.git
cd mlip-pipeline

# Install dependencies using uv
uv sync
```

## Usage

### 1. Initialize Configuration
Generate a default configuration file:
```bash
uv run pyacemaker init --path config.yaml
```

### 2. Configure
Edit `config.yaml` to set up your project. For DFT (Quantum Espresso), ensure you have pseudopotentials available:
```yaml
oracle:
  type: qe
  command: "mpirun -np 4 pw.x"
  pseudo_dir: "/path/to/pseudos"
  pseudopotentials:
    Si: "Si.pbe-n-kjpaw_psl.1.0.0.UPF"
```

### 3. Run Pipeline
Execute the active learning loop:
```bash
uv run pyacemaker run --config config.yaml
```

### 4. Debug Calculation (Single Point)
Run a single DFT calculation on a structure file (xyz, cif, etc.) to verify settings:
```bash
uv run pyacemaker compute --structure structure.xyz --config config.yaml
```

## Architecture

```
src/mlip_autopipec/
├── domain_models/       # Pydantic data schemas (Structure, Config, etc.)
├── infrastructure/      # Concrete implementations
│   ├── generator/       # Structure generators (RandomDisplacement)
│   ├── mocks/           # Mock components for testing
│   └── oracle/          # Physics calculators (DFTManager)
├── interfaces/          # Abstract Base Classes
├── orchestrator/        # Workflow management logic
└── utils/               # Physics and logging utilities
```

## Roadmap

- [x] Cycle 01: Foundation & Orchestrator (Mocks)
- [x] Cycle 02: Data Generation & DFT Integration (Quantum Espresso)
- [ ] Cycle 03: ML Potential Training (PACE/MACE)
- [ ] Cycle 04: MD & Active Learning Strategies
- [ ] Cycle 05: HPC Scalability (Slurm/Parsl)
- [ ] Cycle 06: Production Readiness & Validation

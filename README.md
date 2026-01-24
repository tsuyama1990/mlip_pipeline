# MLIP-AutoPipe: Zero-Human Machine Learning Interatomic Potentials

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Automated Pipeline for Exascale Computing of Machine Learning Interatomic Potentials.**

## Overview

**MLIP-AutoPipe** is an autonomous system designed to democratize the creation of state-of-the-art Machine Learning Interatomic Potentials (MLIPs). By automating the complex cycle of structure generation, First-Principles (DFT) labeling, and active learning, it allows materials scientists to go from chemical composition to production-ready potentials with zero human intervention.

### Why MLIP-AutoPipe?
- **Expert Gap**: Removes the need for deep expertise in ML and DFT workflows.
- **Efficiency**: Uses Active Learning to minimize expensive DFT calculations.
- **Robustness**: Enforces physical constraints (Phonons, Elasticity, ZBL repulsion) to prevent simulation crashes.

## Features

- **Zero-Config Workflow**: Start with a simple `config.yaml` defining your material.
- **Active Learning Loop**: autonomous Exploration -> Detection -> Labeling -> Training cycle.
- **Robust Physics**:
  - **Phonon Stability**: Checks for imaginary modes.
  - **Elasticity**: Validates Born stability criteria.
  - **EOS**: Checks Bulk Modulus.
  - **ZBL Baseline**: Ensures physical core repulsion.
- **Hybrid Engine**: Supports MD (LAMMPS) and kMC (EON).
- **Scalable**: Built for local workstations and HPC clusters (via Dask/Slurm).

## Requirements

- Python 3.11 or higher
- Docker (optional, for containerized execution)
- DFT Code: Quantum Espresso (pw.x) installed and in PATH.
- MD Code: LAMMPS (lmp) installed and in PATH.
- MLIP Engine: Pacemaker or MACE.

## Installation

```bash
git clone https://github.com/your-org/mlip-autopipec.git
cd mlip-autopipec
uv sync
# OR
pip install .
```

## Usage

### 1. Initialize Project
Create a template configuration file:
```bash
mlip-auto init
```

### 2. Configure
Edit `input.yaml` to define your target system (e.g., Al, FeNi):
```yaml
target_system:
  name: "Aluminum"
  elements: ["Al"]
  composition: {"Al": 1.0}
  crystal_structure: "fcc"

dft:
  pseudopotential_dir: "./pseudos"
  command: "pw.x"
```

### 3. Run Workflow
Launch the autonomous active learning loop:
```bash
mlip-auto run loop --config input.yaml
```

### 4. Validation
Validate the physical properties of your trained potential:
```bash
mlip-auto validate input.yaml --phonon --elastic --eos
```

## Architecture

```ascii
src/mlip_autopipec/
├── app.py                      # CLI Entry Point
├── config/                     # Configuration Schemas (Pydantic)
├── generator/                  # Structure Generation (Random, SQS, Defects)
├── dft/                        # DFT Oracle (Quantum Espresso)
├── training/                   # Potential Training (Pacemaker/MACE)
├── inference/                  # MD/kMC Inference (LAMMPS/EON)
├── orchestration/              # Workflow Management
└── validation/                 # Physics Validation Suite (Phonon, Elasticity, EOS)
```

## Roadmap

- [x] Core Framework & Configuration
- [x] Structure Generation (MD/MC/Defects)
- [x] DFT Oracle Interface
- [x] Active Learning Orchestrator
- [x] Validation Suite (Phonon, Elasticity, EOS)
- [ ] Multi-node HPC Support (Slurm/PBS integration)
- [ ] Advanced Phase Diagram Exploration

# PYACEMAKER: Automated MLIP Pipeline

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous system for constructing and operating State-of-the-Art Machine Learning Interatomic Potentials (MLIP) using the Pacemaker (ACE) framework.

## Overview
- **What**: An orchestrator for the "Generation -> Calculation (DFT) -> Training" active learning loop.
- **Why**: Automates the complex workflow of potential development, reducing DFT costs and ensuring robustness.
- **Current Status**: Cycle 01 (Core Framework & Mock Loop) - Ready for testing logic flow.

## Features
- **Core Framework**: Hub-and-Spoke architecture with loose coupling via Protocols.
- **Mock Loop**: Functional orchestration loop with Mock components (Explorer, Oracle, Trainer) to verify workflow logic without external dependencies.
- **Configuration**: Strict Pydantic-based configuration validation.
- **Zero-Config Workflow**: Run the pipeline with a single YAML config.

## Requirements
- Python >= 3.12
- `uv` (recommended) or `pip`

## Installation
```bash
git clone https://github.com/your-org/mlip-pipeline.git
cd mlip-pipeline
uv sync
```

## Usage

### 1. Create a Configuration File
Create `config.yaml`:
```yaml
work_dir: "./workspace"
max_cycles: 3
random_seed: 42
```

### 2. Run the Pipeline
```bash
uv run mlip-pipeline run --config config.yaml
```

Output should show the progress of the active learning cycles:
```text
Starting Active Learning Cycle
--- Starting Cycle 1/3 ---
Generated 1 candidates
MockOracle calculated 1 structures
MockTrainer updated potential using 1 structures
Cycle 1 validation passed
...
```

## Architecture/Structure
```ascii
src/
├── config/             # Configuration schemas
├── domain_models/      # Data structures (Atoms, Dataset)
├── interfaces/         # Core Protocols
├── orchestration/      # Main Loop & Mocks
├── utils/              # Logging & Helpers
└── main.py             # CLI Entry Point
```

## Roadmap
- [x] Cycle 01: Core Framework & Mock Loop
- [ ] Cycle 02: Trainer Module (Pacemaker)
- [ ] Cycle 03: Oracle Module (Quantum Espresso)
- [ ] Cycle 04: Dynamics Engine (LAMMPS)
- [ ] Cycle 05: Adaptive Structure Generation
- [ ] Cycle 06: Validation & Scale-Up

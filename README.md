# PyAceMaker (mlip-autopipec)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

**Democratising MLIP construction using the Pacemaker engine.**

## Overview
**PyAceMaker** is an automated pipeline for constructing Machine Learning Interatomic Potentials (MLIPs). It orchestrates the active learning loop—from structure generation and DFT labeling to potential training—enabling researchers to build robust potentials with minimal manual intervention.

## Features
- **Automated Active Learning Loop**: Seamlessly integrates Exploration, Labeling, and Training stages.
- **Mock Components**: Verify workflow logic without expensive physics simulations (Cycle 01).
- **Configuration Management**: Type-safe YAML configuration using Pydantic.
- **State Persistence**: Robust resume capability with atomic state saving.
- **Rich Logging**: Beautiful console output and detailed file logging.

## Requirements
- Python >= 3.12
- `uv` package manager (recommended) or `pip`

## Installation

```bash
git clone <repository-url>
cd mlip-autopipec
uv sync
```

## Usage

### Initialization
Initialize a new project with a default configuration:

```bash
uv run python src/mlip_autopipec/main.py init --work-dir my_project
```

### Run Loop
Execute the active learning loop:

```bash
uv run python src/mlip_autopipec/main.py run-loop --config-file my_project/config.yaml
```

## Architecture

```ascii
src/mlip_autopipec/
├── components/         # Pluggable components (Generator, Oracle, Trainer)
├── core/               # Core logic (Orchestrator, StateManager, Logger)
├── domain_models/      # Pydantic data models
├── config.py           # Configuration loader
├── constants.py        # Global constants
└── main.py             # CLI entry point
```

## Roadmap
- [x] Cycle 01: Core Framework & Mock Components
- [ ] Cycle 02: Structure Generator (Random/Adaptive)
- [ ] Cycle 03: Oracle Interface (Quantum Espresso)
- [ ] Cycle 04: Trainer Interface (Pacemaker)
- [ ] Cycle 05: Dynamics (LAMMPS)
- [ ] Cycle 06: OTF Loop & Active Learning
- [ ] Cycle 07: Advanced Dynamics (EON)
- [ ] Cycle 08: Validation & Reporting

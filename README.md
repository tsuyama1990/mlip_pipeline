# PYACEMAKER: Automated MLIP Construction System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **"Democratising Machine Learning Interatomic Potentials (MLIP) with Zero-Config Autonomous Agents."**

## Overview

**PYACEMAKER** is an autonomous system designed to construct high-fidelity Machine Learning Interatomic Potentials (MLIPs), specifically based on the Atomic Cluster Expansion (ACE) formalism. It removes the need for deep domain expertise in DFT or machine learning by automating the entire "Active Learning" cycle.

**Why use it?**
*   **Zero-Config:** Start a complex materials discovery campaign with a simple YAML file.
*   **Self-Healing:** Automatically recovers from DFT crashes and simulation failures.
*   **Physics-Aware:** Validates potentials against phonon stability, elastic constants, and equations of state.

## Features

*   **Autonomous Orchestration:** Manages the entire lifecycle of potential generation (Explore -> Label -> Train -> Verify).
*   **Resilient State Management:** Automatically saves progress and resumes from interruptions without data loss.
*   **Configurable Workflows:** Define campaigns using simple, strict YAML configurations.
*   **Mock Mode (Cycle 01):** Validated core framework with mock components for rapid prototyping and flow verification.

## Requirements

*   Python 3.11 or higher
*   `uv` (recommended) or `pip`

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker
uv sync
```

## Usage

### 1. Create a Configuration File

Create a file named `config.yaml`:

```yaml
orchestrator:
  work_dir: "my_simulation_run"
  max_cycles: 5
```

### 2. Run the System

Execute the runner:

```bash
uv run mlip-runner run config.yaml
```

### 3. Output

The system will create a directory `my_simulation_run/` containing:
*   `mlip.log`: Detailed execution logs.
*   `workflow_state.json`: Current state of the active learning loop (auto-updated).

## Architecture

```text
src/mlip_autopipec/
├── main.py                     # CLI Entry Point
├── core/
│   ├── orchestrator.py         # Main Control Loop
│   ├── state_manager.py        # Persistence Logic
│   └── logger.py               # Logging Setup
├── domain_models/
│   ├── config.py               # Pydantic Schemas
│   ├── datastructures.py       # State Models
│   └── enums.py                # Status Codes
└── utils/
    └── io.py                   # File Helpers
```

## Roadmap

*   **Cycle 02:** Structure Generation (Adaptive Exploration)
*   **Cycle 03:** Oracle Integration (DFT/Quantum Espresso)
*   **Cycle 04:** Training Engine (Pacemaker)
*   **Cycle 05:** MD Dynamics (LAMMPS)

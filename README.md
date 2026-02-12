# PYACEMAKER: Automated MLIP Construction & Operation System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an end-to-end automated system for constructing "State-of-the-Art" Machine Learning Interatomic Potentials (MLIP) using the Atomic Cluster Expansion (ACE) formalism. It empowers materials scientists to go from **Zero to Simulation** without writing code, automating the complex "Explore-Label-Train" active learning cycle.

## 🚀 Overview

*   **What**: An automated pipeline to train, validate, and run MLIPs.
*   **Why**: Manual training is error-prone and inefficient. PYACEMAKER automates the entire loop.
*   **Key Tech**: ACE (Pacemaker), Active Learning (D-Optimality), Hybrid Potentials (ZBL/LJ), Adaptive Kinetic Monte Carlo (aKMC).

## ✨ Features (Current Status)

*   **Zero-Config Workflow**: Define your material and goals in a single `config.yaml`.
*   **Robust Configuration**: Strict validation of all inputs using Pydantic schemas.
*   **State Management**: Atomic state saving ensures workflows can be paused and resumed safely.
*   **Centralized Logging**: Comprehensive logging for all pipeline activities.
*   **CLI Interface**: Easy-to-use command line interface (`mlip-runner`).

## 🛠️ Prerequisites

*   **Python**: 3.12 or higher
*   **Package Manager**: `uv` (highly recommended) or `pip`

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    We use `uv` for fast, reproducible environments.
    ```bash
    uv sync
    ```

## 🚀 Usage

### Initialize a Project
Generate a default configuration file (`config.yaml`) to start your project.

```bash
uv run mlip-runner init
```

### Run the Pipeline
Execute the automated workflow using your configuration.

```bash
uv run mlip-runner run config.yaml
```

Example `config.yaml`:
```yaml
orchestrator:
  work_dir: mlip_run
  max_iterations: 10
generator:
  type: RANDOM
  num_structures: 10
oracle:
  type: QUANTUM_ESPRESSO
  command: pw.x
  mixing_beta: 0.7
trainer:
  type: PACEMAKER
  r_cut: 5.0
  max_deg: 3
```

## 🏗️ Architecture

The system follows a modular architecture orchestrated by a central controller.

```ascii
src/mlip_autopipec/
├── core/               # Config, Logging, State Management
├── domain_models/      # Pydantic Schemas & Data Structures
├── main.py             # CLI Entry Point
└── ...                 # Future modules (Generator, Oracle, Trainer, etc.)
```

## 💻 Development

We enforce strict code quality standards.

**Run Tests:**
```bash
uv run pytest
```

**Run Linters:**
```bash
uv run ruff check .
uv run mypy .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

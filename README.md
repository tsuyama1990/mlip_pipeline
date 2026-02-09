# PyAceMaker: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PyAceMaker** is a comprehensive, automated system designed to democratize the creation of Machine Learning Interatomic Potentials (MLIPs).

## Overview

### What is PyAceMaker?
PyAceMaker (MLIP AutoPipeline) is an orchestration framework that automates the entire lifecycle of developing atomic cluster expansion (ACE) potentials. From initial random structure generation to active learning loops involving molecular dynamics and DFT calculations.

### Why use it?
Building MLIPs traditionally requires manual hand-holding of DFT calculations, fitting procedures, and validation steps. PyAceMaker provides a "Zero-Config" workflow where a single YAML file drives the entire process, ensuring reproducibility and efficiency.

## Features

*   **Core Orchestrator**: Centralized management of the active learning loop.
*   **Schema-First Configuration**: Strict validation of configuration files using Pydantic V2.
*   **Modular Architecture**: Extensible components for Generator, Oracle, Trainer, Dynamics, and Validator.
*   **Logging System**: Comprehensive logging to both console and file for audit trails.
*   **Mock Components**: Built-in mock implementations for testing the pipeline flow without external dependencies (DFT/LAMMPS).

## Requirements

*   **Python 3.12+**
*   **uv** (recommended) or pip

## Installation

```bash
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker
uv sync
```

## Usage

### Basic Execution

To run the pipeline with a configuration file:

```bash
uv run python src/mlip_autopipec/main.py config.yaml
```

### Configuration Example (`config.yaml`)

```yaml
orchestrator:
  work_dir: "./my_experiment"
  max_cycles: 5
  uncertainty_threshold: 5.0

generator:
  type: "random"
  seed: 42

oracle:
  type: "mock"

trainer:
  type: "mock"

dynamics:
  type: "mock"

validator:
  type: "mock"
```

## Architecture

The project is structured as follows:

```ascii
src/mlip_autopipec/
├── components/       # Abstract Base Classes and Mock implementations
├── core/             # Orchestrator and Logging logic
├── domain_models/    # Pydantic Configuration Schemas
├── constants.py      # System-wide constants
└── main.py           # CLI Entry Point
```

## Roadmap

*   **Cycle 02**: Structure Generator (Adaptive & Random)
*   **Cycle 03**: Oracle (DFT with Quantum Espresso)
*   **Cycle 04**: Trainer (Pacemaker Integration)
*   **Cycle 05**: Dynamics (LAMMPS Integration)
*   **Cycle 06**: Active Learning Loop (OTF)
*   **Cycle 07**: Advanced Dynamics (EON/kMC)
*   **Cycle 08**: Validator (Phonons, EOS, Elasticity)

# MLIP AutoPipeC

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)

> Automated construction of Machine Learning Interatomic Potentials (MLIP) using Active Learning.

## Overview

**What:** MLIP AutoPipeC is an orchestration framework that automates the generation of interatomic potentials. It integrates Structure Generation, First-Principles Calculations (DFT), and Machine Learning Potential Training (ACE) into a seamless, self-correcting loop.

**Why:** Creating robust MLIPs manually is error-prone and labor-intensive. This tool ensures D-Optimality (data efficiency) and physical stability (robustness) by automating the active learning cycle, allowing researchers to focus on materials science rather than workflow management.

## Features

*   **Zero-Config Workflow**: Initialize complex pipelines with a single `config.yaml`.
*   **Robust State Management**: Automatic state persistence ensures jobs can be resumed after interruptions.
*   **Modular Architecture**: Plug-and-play components for Structure Generation, Oracle (DFT), and Training.
*   **Mock Mode**: Skeleton execution mode for rapid development and testing without expensive physics backends.
*   **Strict Validation**: Pydantic-based configuration ensures fail-fast behavior for invalid inputs.

## Requirements

*   Python 3.12+
*   `uv` (recommended) or `pip`
*   External Dependencies (Optional for Cycle 01):
    *   `pace_train` (Pacemaker)

## Installation

```bash
git clone https://github.com/your-org/mlip-autopipec.git
cd mlip-autopipec
uv sync
```

## Usage

1.  **Prepare Configuration**: Create a `config.yaml` file.

    ```yaml
    project:
      name: "TitaniumOxide"
      seed: 42

    training:
      dataset_path: "/path/to/data.pckl"
      max_epochs: 100

    orchestrator:
      max_iterations: 10
    ```

2.  **Run the Pipeline**:

    ```bash
    uv run python -m mlip_autopipec.main config.yaml
    ```

3.  **Output**:
    *   `state.json`: Tracks the progress of the active learning loop.
    *   `output.yace`: The trained potential (in production mode).
    *   Logs are printed to stdout (and optionally to file).

## Architecture

```
src/mlip_autopipec/
├── config/             # Pydantic Configuration Models
├── domain_models/      # Core Data Structures (WorkflowState, Potential)
├── orchestration/      # State Machine & Main Loop
├── physics/            # Interfaces for Physics Engines (Training, DFT, MD)
└── utils/              # Shared Utilities (File Ops, Logging)
```

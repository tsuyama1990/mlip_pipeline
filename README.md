# PyAceMaker: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PyAceMaker** is a comprehensive, automated system designed to democratize the creation of Machine Learning Interatomic Potentials (MLIPs). Leveraging the power of the **Pacemaker** (Atomic Cluster Expansion) engine, it enables researchers to go from a simple chemical composition to a "State-of-the-Art" potential with zero manual configuration.

Bridging the gap between high-accuracy DFT and large-scale MD, PyAceMaker automates the tedious cycle of structure generation, labeling, training, and validation, ensuring both data efficiency and physical robustness.

## 🚀 Key Features

*   **Zero-Config Workflow**: Define your material system in a single YAML file and let the Orchestrator handle the rest.
*   **Active Learning Loop**: Intelligently samples the configuration space using uncertainty quantification ($\gamma$), minimizing expensive DFT calculations.
*   **Physics-Informed Robustness**: Implements "Delta Learning" with a physical baseline (ZBL/LJ) to ensure stability in high-energy regimes and prevent simulation crashes.
*   **Time-Scale Bridging**: Seamlessly integrates Molecular Dynamics (LAMMPS) for fast kinetics and Adaptive Kinetic Monte Carlo (EON) for rare events like diffusion and ordering.
*   **Automated Validation**: Rigorous quality assurance with built-in phonon dispersion, elastic constant, and EOS calculations.

## 🏗️ Architecture Overview

PyAceMaker follows a modular, orchestrator-based architecture. A central "Brain" coordinates specialized agents for exploration, labeling, training, and execution.

```mermaid
graph TD
    User[User Config YAML] --> Orch[Orchestrator]

    subgraph "Active Learning Loop"
        Orch -->|1. Explore| Dyn[Dynamics Engine<br/>(LAMMPS / EON)]
        Dyn -->|Stream Structures| Gen[Structure Generator<br/>(Adaptive Policy)]
        Gen -->|2. Generate Candidates| Cand[Candidate Pool]

        Cand -->|3. Select (D-Opt)| Select[Active Set Selector]
        Select -->|Selected Structures| Oracle[Oracle<br/>(QE / VASP)]

        Oracle -->|4. Compute Labels| Data[Labeled Dataset]

        Data -->|5. Train| Trainer[Trainer<br/>(Pacemaker)]
        Trainer -->|New Potential| Dyn
    end

    Trainer -->|Final Potential| Val[Validator]
    Val -->|Report| Report[HTML Report]
```

## 🛠️ Prerequisites

*   **Python 3.12+**
*   **uv** (Recommended for dependency management)
*   **LAMMPS** (compiled with `PACE` package)
*   **Quantum Espresso** (`pw.x`) or **VASP**
*   **Pacemaker** (`pace_train`, `pace_activeset`)

## 📦 Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Environment Setup**:
    Copy the example environment file and configure your paths.
    ```bash
    cp .env.example .env
    # Edit .env to point to your LAMMPS and QE executables
    ```

## ⚡ Usage

### Quick Start
To train a potential for the Fe-Pt system on MgO:

1.  **Prepare Configuration**:
    Edit `config.yaml` to specify your target system.
    ```yaml
    project_name: "FePt_MgO"
    generator:
      type: "adaptive"
      composition: "FePt"
    oracle:
      type: "qe"
      command: "mpirun -np 4 pw.x"
    ```

2.  **Run the Pipeline**:
    ```bash
    uv run python main.py --config config.yaml
    ```

3.  **Monitor Progress**:
    The system will log its progress to `mlip_pipeline.log`. You can watch the active learning loop in real-time.

## 💻 Development Workflow

We follow a strict 8-cycle development plan.

*   **Running Tests**:
    ```bash
    uv run pytest
    ```
*   **Linting**:
    ```bash
    uv run ruff check .
    uv run mypy .
    ```

## 📂 Project Structure

```ascii
src/mlip_pipeline/
├── core/               # Orchestrator, Config, Logging
├── components/         # Pluggable Modules
│   ├── generators/     # Structure Generation
│   ├── oracles/        # DFT Interface
│   ├── trainers/       # Pacemaker Interface
│   ├── dynamics/       # LAMMPS/EON Interface
│   └── validators/     # Physics Validation
└── domain_models/      # Pydantic Data Models
dev_documents/          # Specs and Documentation
tests/                  # Unit and Integration Tests
tutorials/              # Jupyter Notebooks
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

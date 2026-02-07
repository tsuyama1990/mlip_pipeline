# PYACEMAKER: Autopilot for Machine Learning Potentials

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous system that democratizes the creation of State-of-the-Art Machine Learning Interatomic Potentials (MLIPs). Currently in **Cycle 01**, it provides a robust "Walking Skeleton" architecture with a complete active learning loop simulation using mock components.

## 🚀 Key Features

-   **Active Learning Orchestration**: Fully automated loop: Generate -> Oracle -> Train -> Dynamics.
-   **Mock Infrastructure**: Simulate the entire pipeline without expensive physics calculations to verify logic and flow.
-   **Strict Typing & Validation**: Built on Pydantic and Abstract Base Classes for reliability.
-   **Configurable**: Simple YAML configuration to control all components.

## 🛠 Prerequisites

-   **Python 3.12+**
-   **uv** (Fast Python package manager)

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install Dependencies**
    ```bash
    uv sync
    ```

3.  **Activate Environment** (Optional if using `uv run`)
    ```bash
    source .venv/bin/activate
    ```

## 🏃 Usage

### 1. Initialize Configuration
Generate a default configuration file (`config.yaml`) with mock settings.
```bash
uv run pyacemaker init
```

### 2. Run the Pipeline
Execute the active learning loop simulation.
```bash
uv run pyacemaker run --config config.yaml
```

You should see logs indicating the progression of cycles (Generation, Oracle computation, Training, Dynamics).

## 🏗 Architecture

The system follows a modular architecture:

```
src/mlip_autopipec/
├── domain_models/    # Pydantic data schemas (Structure, Config)
├── interfaces/       # Abstract Base Classes (Oracle, Trainer, etc.)
├── infrastructure/   # Concrete implementations (Mocks)
├── orchestrator/     # Main logic loop
└── main.py           # CLI entry point
```

## 🗺 Roadmap

-   **Cycle 01**: Walking Skeleton & Mocks (Completed)
-   **Cycle 02**: Real Physics Integrations (QE, Pacemaker)
-   **Cycle 03**: Advanced Dynamics & Exploration
-   ...

## 📄 License

This project is licensed under the MIT License.

# PyAceMaker: Automated MLIP Construction System

![Status](https://img.shields.io/badge/status-active_development-blue)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PyAceMaker** is a fully automated, "Zero-Config" system for constructing and operating Machine Learning Interatomic Potentials (MLIP) using the **Pacemaker** (Atomic Cluster Expansion) engine. It empowers materials scientists to generate "State-of-the-Art" potentials with minimal manual intervention.

---

## 🚀 Features

*   **Zero-Config Workflow**: Initiate a complete training pipeline with a single YAML file.
*   **Modular Architecture**: Components (Generator, Oracle, Trainer, Dynamics, Validator) are decoupled and configurable.
*   **Robust Configuration**: Strict validation of configuration files using Pydantic.
*   **Mocking Support**: Built-in mock components for testing the pipeline logic without external dependencies.
*   **Structured Logging**: Comprehensive logging for traceability.

## 🛠️ Prerequisites

*   **Python 3.12+**
*   **uv** (Recommended) or `pip`

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    ```bash
    uv sync
    ```
    (Or `pip install -e .`)

3.  **Configuration**
    Copy the example configuration:
    ```bash
    cp config.example.yaml config.yaml
    ```

## ⚡ Usage

### Run the Pipeline
To initialize the orchestrator and verify configuration:

```bash
uv run python -m mlip_autopipec.main --config config.yaml
```

### Run Tests
```bash
uv run pytest
```

## 📂 Project Structure

```
.
├── src/
│   └── mlip_autopipec/
│       ├── core/           # Orchestrator & Logging
│       ├── components/     # Base Classes & Mock Implementations
│       ├── domain_models/  # Pydantic Data Models
│       ├── config.py       # Configuration Schema
│       ├── factory.py      # Component Factory
│       └── main.py         # Entry Point
├── tests/                  # Unit & UAT tests
├── config.example.yaml     # Example configuration
└── pyproject.toml          # Project metadata
```

## 📄 License

MIT License.

# PyAceMaker: Automated Machine Learning Interatomic Potential Construction

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**PyAceMaker** (`mlip-pipeline`) is a "Zero-Config" autonomous system designed to democratise the creation of State-of-the-Art Machine Learning Interatomic Potentials (MLIPs). By orchestrating `Pacemaker`, `QuantumEspresso`, `LAMMPS`, and `EON`, it allows researchers to generate robust, physics-informed potentials for complex materials.

## Features (Current Release)

-   **Unified Configuration**: A single `config.yaml` validated by Pydantic V2.
-   **Autonomous Orchestrator**: Manages workflow state and component lifecycle.
-   **Mock Capabilities**: Includes mock components for pipeline verification without external binaries.
-   **Robust Logging**: Centralized logging system with file rotation and console output.

## Prerequisites

-   **Python 3.12+**
-   **uv** (Recommended package manager)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/mlip_autopipec.git
    cd mlip_autopipec
    ```

2.  **Install dependencies:**
    ```bash
    uv sync
    ```

## Usage

### 1. Create a Configuration File
Create a `config.yaml` file (see `config.example.yaml` for reference):

```yaml
project_name: "MyFirstProject"
orchestrator:
  work_dir: "experiments/run_01"
  max_iterations: 10
  state_file: "workflow_state.json"
generator:
  type: "MOCK"
  enabled: true
oracle:
  type: "MOCK"
  enabled: true
trainer:
  type: "MOCK"
  enabled: false
dynamics:
  type: "MOCK"
  enabled: false
```

### 2. Run the Pipeline
Execute the CLI using `uv run`:

```bash
uv run mlip-runner run config.yaml
```

The system will validate the configuration, initialize the orchestrator, and run the workflow (currently in mock mode).

### 3. Check Outputs
Inspect the logs and state file in the working directory:
```bash
cat experiments/run_01/mlip_pipeline.log
cat experiments/run_01/workflow_state.json
```

## Architecture

```ascii
src/mlip_autopipec/
├── components/      # Worker modules (Generator, Oracle, Trainer, etc.)
├── core/            # Logic (Orchestrator, State Manager, Config)
├── domain_models/   # Pydantic data models
└── main.py          # CLI Entry point
```

## License

MIT License. See [LICENSE](LICENSE) for details.

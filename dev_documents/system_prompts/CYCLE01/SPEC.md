# Cycle 01 Specification: System Skeleton & Mock Infrastructure

## 1. Summary
The primary objective of Cycle 01 is to establish the "Walking Skeleton" of the PYACEMAKER system. We will implement the core architectural components—the Orchestrator, Abstract Base Classes (Interfaces), and Configuration Management—along with a complete set of "Mock" implementations for the heavy-lifting modules (Oracle, Trainer, Dynamics). This cycle ensures that the control flow, data passing, and error handling logic are robust before we introduce the complexities of real physics engines (DFT, MD). By the end of this cycle, we will have a CLI tool that can read a YAML configuration and simulate a full Active Learning loop, producing logs and dummy artifacts.

## 2. System Architecture

### File Structure
We will create the directory structure defined in the `SYSTEM_ARCHITECTURE.md`. Files to be created in this cycle are marked in **bold**.

```
mlip-pipeline/
├── **pyproject.toml**              # Dependency and build management
├── **README.md**                   # Entry point documentation
├── **config.yaml**                 # Default configuration template
├── src/
│   └── mlip_autopipec/
│       ├── **__init__.py**
│       ├── **main.py**             # CLI Entry point (Typer)
│       ├── domain_models/          # Pydantic models
│       │   ├── **__init__.py**
│       │   ├── **config.py**       # GlobalConfig
│       │   ├── **structure.py**    # Structure, Dataset
│       │   └── **potential.py**    # Potential
│       ├── interfaces/             # Abstract Base Classes
│       │   ├── **__init__.py**
│       │   ├── **oracle.py**       # BaseOracle
│       │   ├── **trainer.py**      # BaseTrainer
│       │   ├── **dynamics.py**     # BaseDynamics
│       │   └── **generator.py**    # BaseStructureGenerator
│       ├── infrastructure/         # Concrete Implementations
│       │   ├── **__init__.py**
│       │   ├── **mocks.py**        # MockOracle, MockTrainer, etc.
│       │   ├── oracle/
│       │   ├── trainer/
│       │   ├── dynamics/
│       │   └── generator/
│       ├── orchestrator/           # Logic wiring
│       │   ├── **__init__.py**
│       │   └── **simple_orchestrator.py**
│       └── utils/
│           ├── **__init__.py**
│           └── **logging.py**      # Logging configuration
└── tests/
    ├── **__init__.py**
    ├── unit/
    │   ├── **test_config.py**
    │   └── **test_orchestrator.py**
    └── integration/
        └── **test_mock_pipeline.py**
```

## 3. Design Architecture

### Domain Models (`domain_models/`)
We use Pydantic V2 for all data validation.

-   **`GlobalConfig`**: The root configuration object.
    -   `project_name`: str
    -   `seed`: int
    -   `oracle`: OracleConfig (discriminator: "mock", "qe", "vasp")
    -   `trainer`: TrainerConfig (discriminator: "mock", "pacemaker")
    -   `dynamics`: DynamicsConfig (discriminator: "mock", "lammps")
    -   `generator`: GeneratorConfig (discriminator: "mock", "random")

-   **`Structure`**: A simplified wrapper for atomic data.
    -   `positions`: np.ndarray (N, 3)
    -   `cell`: np.ndarray (3, 3)
    -   `species`: List[str]
    -   `energy`: Optional[float]
    -   `forces`: Optional[np.ndarray]

### Interfaces (`interfaces/`)
All components must inherit from these ABCs.

-   **`BaseOracle`**:
    -   `compute(structure: Structure) -> Structure`: Adds energy/forces.
-   **`BaseTrainer`**:
    -   `train(dataset: Dataset, params: Dict) -> Path`: Returns path to `.yace` file.
-   **`BaseDynamics`**:
    -   `run(potential: Path, structure: Structure) -> ExplorationResult`: Returns trajectory and halt status.
-   **`BaseStructureGenerator`**:
    -   `generate(base_structure: Structure, strategy: str) -> List[Structure]`: Proposes new candidates.

### Mock Implementations (`infrastructure/mocks.py`)
-   **`MockOracle`**: Randomly assigns energy and forces to the input structure. Sleeps for 0.1s to simulate work.
-   **`MockTrainer`**: Creates a dummy file `potential.yace` and returns its path.
-   **`MockDynamics`**: Returns a random `ExplorationResult` (sometimes "halted", sometimes "converged").

## 4. Implementation Approach

1.  **Setup Project**: Initialize `uv`, `ruff`, `mypy` in `pyproject.toml`.
2.  **Define Models**: Implement `domain_models/*.py`. Ensure strict type checking.
3.  **Define Interfaces**: Create the ABCs in `interfaces/*.py`.
4.  **Implement Mocks**: Create `infrastructure/mocks.py` implementing the interfaces.
5.  **Implement Orchestrator**: Write the main loop in `orchestrator/simple_orchestrator.py`.
    -   Load Config.
    -   Instantiate components based on config type (Factory pattern).
    -   Run the loop: Generate -> Oracle -> Train -> Dynamics -> Check Halt.
6.  **CLI**: Implement `main.py` using `typer`.
    -   Command: `run --config config.yaml`
    -   Command: `init` (creates a sample config)

## 5. Test Strategy

### Unit Testing (`tests/unit/`)
-   **`test_config.py`**: Verify that `GlobalConfig` correctly parses a valid YAML and rejects an invalid one (e.g., missing fields).
-   **`test_orchestrator.py`**: Mock the components *again* (using `unittest.mock`) to verify the Orchestrator calls them in the correct order (Generate -> Compute -> Train).

### Integration Testing (`tests/integration/`)
-   **`test_mock_pipeline.py`**:
    -   Create a temporary `config.yaml` setting all components to "mock".
    -   Run the full `main.py` entry point.
    -   Assert that `potential.yace` is created.
    -   Assert that logs contain "MockOracle computed structure".
    -   Assert that the process exits with code 0.

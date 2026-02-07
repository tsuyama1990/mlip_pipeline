# Cycle 01 Specification: Core Framework & Mocks

## 1. Summary
This cycle establishes the foundational architecture of the PYACEMAKER system. The goal is to create the "skeleton" of the application: the configuration system, the abstract interfaces for all components, the domain models (Pydantic), and "Mock" implementations of these components. By the end of this cycle, we will have an executable `Orchestrator` that can run a full "Active Learning Loop" using fake data, verifying the logic flow without requiring external physics binaries.

## 2. System Architecture

### 2.1. File Structure
The following files must be created. **Bold** files are the focus of this cycle.

src/mlip_autopipec/
├── **__init__.py**
├── **main.py**                     # CLI Entrypoint (Typer)
├── **factory.py**                  # Component Factory
├── domain_models/
│   ├── **__init__.py**
│   ├── **config.py**               # GlobalConfig & Component Configs
│   ├── **structure.py**            # Structure & Dataset Models
│   ├── **potential.py**            # Potential & ExplorationResult Models
│   └── **validation.py**           # ValidationResult Model
├── interfaces/
│   ├── **__init__.py**
│   ├── **oracle.py**               # BaseOracle ABC
│   ├── **trainer.py**              # BaseTrainer ABC
│   ├── **dynamics.py**             # BaseDynamics ABC
│   ├── **generator.py**            # BaseStructureGenerator ABC
│   ├── **validator.py**            # BaseValidator ABC
│   └── **selector.py**             # BaseSelector ABC
├── infrastructure/
│   ├── **__init__.py**
│   └── **mocks.py**                # Mock implementations for all interfaces
└── orchestrator/
    ├── **__init__.py**
    └── **simple_orchestrator.py**  # The main loop logic

### 2.2. Class Diagram
*   `SimpleOrchestrator` depends on `BaseOracle`, `BaseTrainer`, `BaseDynamics`, etc.
*   `MockOracle` implements `BaseOracle`.
*   `GlobalConfig` contains sub-configs (`OracleConfig`, `TrainerConfig`, etc.).

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/`)
All data exchange must use strict Pydantic models.

*   **`Structure`**: Represents an atomic configuration.
    *   Fields: `atoms` (ASE Atoms object, serialized via custom validator), `energy` (float, optional), `forces` (array, optional), `stress` (array, optional).
    *   *Note*: Pydantic v2 `ArbitraryTypesAllowed` is needed for ASE objects.
*   **`GlobalConfig`**: The root configuration.
    *   Uses **Discriminated Unions** to select component types (e.g., `oracle: Union[MockOracleConfig, QEScConfig] = Field(..., discriminator='type')`).
    *   This ensures that if the user sets `type: mock`, only mock-related parameters are validated.

### 3.2. Interfaces (`interfaces/`)
All components must inherit from Abstract Base Classes (ABCs).

*   **`BaseOracle`**: `compute(structures: Iterable[Structure]) -> Iterator[Structure]`
*   **`BaseTrainer`**: `train(dataset: Iterable[Structure], ...) -> Potential`
*   **`BaseDynamics`**: `run(potential: Potential, start_structure: Structure) -> ExplorationResult`
    *   `ExplorationResult` contains: `final_structure`, `trajectory_path`, `status` (CONVERGED/HALTED), `max_uncertainty`.

### 3.3. Mocks (`infrastructure/mocks.py`)
Mock components must simulate realistic behavior:
*   `MockOracle`: Adds random noise to energy/forces instead of running DFT.
*   `MockTrainer`: Creates a dummy `.yace` file and returns a `Potential` object pointing to it.
*   `MockDynamics`: Returns a slightly perturbed structure and randomly triggers a "High Uncertainty" halt based on a probability parameter.

### 3.4. Orchestrator (`simple_orchestrator.py`)
The logic core.
1.  Initialize components via `factory.py` using `GlobalConfig`.
2.  Loop `max_cycles` times.
3.  Call `Dynamics.run()`.
4.  If HALTED: Call `Selector` -> `Oracle` -> `Trainer`.
5.  Call `Validator`.

## 4. Implementation Approach

1.  **Setup Project**: Initialize `pyproject.toml` (already done) and directory structure.
2.  **Define Domain Models**: Create `domain_models/*.py`. Ensure ASE compatibility.
3.  **Define Interfaces**: Create `interfaces/*.py`.
4.  **Implement Mocks**: Create `infrastructure/mocks.py`. Implement logging to trace calls.
5.  **Implement Factory**: Create `factory.py` to instantiate classes based on config strings.
6.  **Implement Orchestrator**: Write the loop in `simple_orchestrator.py`.
7.  **Implement CLI**: Create `main.py` using `Typer` to load config and run orchestrator.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Test**: Verify `GlobalConfig` parses valid YAML and rejects invalid ones.
*   **Factory Test**: Verify `create_oracle({'type': 'mock'})` returns a `MockOracle` instance.
*   **Mock Test**: Verify `MockOracle.compute()` adds forces to structures.

### 5.2. Integration Testing
*   **Loop Test**: Run the `SimpleOrchestrator` with a fully mocked config.
    *   *Success Criteria*: The loop runs for `N` cycles, "generating" potentials and "running" dynamics without crashing. Logs show flow from Dynamics -> Oracle -> Trainer.

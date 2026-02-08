# Cycle 01 Specification: Core Framework & Mocks

## 1. Summary

The objective of Cycle 01 is to establish the **architectural skeleton** of the PYACEMAKER system. We will not implement any "real" scientific logic (DFT or MD) in this cycle. Instead, we focus on creating a robust, type-safe foundation that orchestrates the data flow between components. We will define the **Abstract Base Classes (ABCs)** that contractually oblige future components to adhere to specific interfaces. We will also implement **Mock Components** that simulate the behavior of the real tools (logging their actions instead of calculating), allowing us to verify the `Orchestrator`'s logic and the CLI's functionality immediately.

## 2. System Architecture

We will create the directory structure and the core files. Files in **bold** are to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── __init__.py
├── **constants.py**          # Global constants (defaults)
├── **factory.py**            # Component Factory
├── **main.py**               # CLI Entry point
├── components/
│   ├── **dynamics/**
│   │   ├── __init__.py
│   │   └── **mock.py**       # Mock Dynamics
│   ├── **generator/**
│   │   ├── __init__.py
│   │   └── **mock.py**       # Mock Generator
│   ├── **oracle/**
│   │   ├── __init__.py
│   │   └── **mock.py**       # Mock Oracle
│   ├── **trainer/**
│   │   ├── __init__.py
│   │   └── **mock.py**       # Mock Trainer
│   └── **validator/**
│       ├── __init__.py
│       └── **mock.py**       # Mock Validator
├── core/
│   ├── __init__.py
│   ├── **dataset.py**        # Basic Dataset skeleton
│   └── **orchestrator.py**   # Main loop logic
├── domain_models/
│   ├── __init__.py
│   ├── **config.py**         # Pydantic Config Models
│   ├── **structure.py**      # Pydantic Structure Model
│   └── **potential.py**      # Pydantic Potential Model
├── infrastructure/
│   ├── __init__.py
│   └── **mocks.py**          # (Optional) Shared mock utils
├── interfaces/
│   ├── __init__.py
│   ├── **base_dynamics.py**
│   ├── **base_generator.py**
│   ├── **base_oracle.py**
│   ├── **base_trainer.py**
│   └── **base_validator.py**
└── utils/
    ├── __init__.py
    └── **logging.py**        # Structured logging setup
```

## 3. Design Architecture

### 3.1. Pydantic Domain Models (`domain_models/`)
We define the data structures that flow through the system.
-   **`GlobalConfig`**: The root configuration.
    -   Must contain `workdir`, `max_cycles`.
    -   Must contain subsections: `generator`, `oracle`, `trainer`, `dynamics`.
    -   Each subsection has a `type` field (e.g., "mock", "lammps").
-   **`Structure`**: A wrapper around atomic data.
    -   Fields: `positions` (Nx3 float), `atomic_numbers` (N int), `cell` (3x3 float), `pbc` (3 bool).
    -   Optional: `forces`, `stress`, `energy`, `properties` (dict).
-   **`Potential`**: Metadata.
    -   Fields: `path` (Path), `version` (str), `metrics` (dict).

### 3.2. Interfaces (`interfaces/`)
We define ABCs using `abc.ABC`.
-   **`BaseGenerator`**: `generate(potential: Potential | None) -> list[Structure]`
-   **`BaseOracle`**: `compute(structures: list[Structure]) -> list[Structure]` (Returns labeled structures)
-   **`BaseTrainer`**: `train(dataset: Dataset, initial_potential: Potential | None) -> Potential`
-   **`BaseDynamics`**: `run(potential: Potential) -> list[Structure]` (Returns high-uncertainty structures)
-   **`BaseValidator`**: `validate(potential: Potential) -> dict`

### 3.3. Orchestrator Logic
The `Orchestrator` is a class that:
1.  Loads `GlobalConfig`.
2.  Uses `factory.py` to instantiate components based on config.
3.  Runs a loop `for cycle in range(max_cycles)`:
    -   Calls `Generator` or `Dynamics` to get candidates.
    -   Calls `Oracle` to label them.
    -   Updates `Dataset`.
    -   Calls `Trainer` to update `Potential`.
    -   Calls `Validator` to check quality.

## 4. Implementation Approach

1.  **Setup Environment**: Initialize `src/mlip_autopipec` package.
2.  **Domain Models**: Implement `structure.py` and `config.py` using Pydantic.
3.  **Interfaces**: Define the ABCs in `interfaces/`.
4.  **Mocks**: Implement `MockGenerator`, `MockOracle`, etc., in `components/*/mock.py`.
    -   *Behavior*: They should print/log "MockGenerator: Generated 5 structures" and return dummy objects.
5.  **Factory**: Implement `create_component(type: str, config: dict)` in `factory.py`.
6.  **Orchestrator**: Implement the main loop in `core/orchestrator.py`.
7.  **CLI**: Implement `main.py` using `typer` (or `argparse`). It should accept `run <config_path>`.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Config Validation**: Create valid and invalid YAMLs. Ensure `GlobalConfig` raises `ValidationError` on missing fields.
-   **Factory**: Ensure `factory.create_generator({"type": "mock"})` returns an instance of `MockGenerator`.

### 5.2. Integration Testing (The "Walking Skeleton")
-   **Scenario**: Run the full pipeline with a "Mock Config".
-   **Input**: `config_mock.yaml`
    ```yaml
    workdir: "runs/test_run"
    max_cycles: 2
    generator:
      type: "mock"
    oracle:
      type: "mock"
    trainer:
      type: "mock"
    dynamics:
      type: "mock"
    ```
-   **Expected Output**: The CLI should run without error. Logs should show the sequence: Gen -> Oracle -> Train -> Dyn -> Val (repeated 2 times).

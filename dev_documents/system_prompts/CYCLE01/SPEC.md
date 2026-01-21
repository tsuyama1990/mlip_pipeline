# Cycle 01 Specification: Foundation & Configuration

## 1. Summary

Cycle 01 lays the definitive groundwork for the MLIP-AutoPipe system. The primary objective of this cycle is to establish the core software infrastructure.

This cycle focuses on three key technical pillars:
1.  **Strict Configuration Management**: We will define the "Source of Truth" for the entire pipeline using **Pydantic V2**. `MLIPConfig` will be the root configuration object.
2.  **Persistence Layer (Database)**: We will establish the interface to the database using `ase.db` wrapped in `DatabaseManager`.
3.  **Command Line Interface (CLI)**: We will create the entry point for the application using **Typer**.

## 2. System Architecture

### File Structure
```
mlip_autopipec/
├── **__init__.py**
├── **app.py**                  # CLI Entrypoint (Typer App)
├── config/
│   ├── **__init__.py**
│   ├── **models.py**           # Aggregated Configuration Model (MLIPConfig, RuntimeConfig)
│   └── schemas/
│       ├── **__init__.py**
│       ├── **common.py**       # Shared Enums and Types (TargetSystem)
│       ├── **dft.py**          # DFT Configuration Schema (DFTConfig)
│       ├── **training.py**     # Training Configuration Schema
│       └── **inference.py**    # Inference Configuration Schema
├── core/
│   ├── **__init__.py**
│   ├── **database.py**         # DatabaseManager (ASE-db wrapper)
│   ├── **logging.py**          # Centralized Logging Setup (Rich integration)
│   └── **services.py**         # Application Services (e.g., Config Validation)
└── data_models/
    ├── **__init__.py**
    └── **common.py**           # Domain Data Types (Enums like Status)
```

### Data Dictionary

| Model Name | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| **TargetSystem** | elements | List[str] | List of chemical symbols (e.g., `["Fe", "Ni"]`). |
| | composition | Dict[str, float] | Atomic fractions (e.g., `{"Fe": 0.7, "Ni": 0.3}`). |
| | crystal_structure | Optional[str] | Base structure (e.g., "fcc", "bcc", "hcp"). |
| **DFTConfig** | pseudopotential_dir | Path | Directory containing .UPF files. |
| | ecutwfc | float | Wavefunction cutoff energy (Ry). |
| | kspacing | float | Inverse K-point density (1/A). |
| | nspin | int | Spin polarization (1=off, 2=on). |
| **MLIPConfig** | target_system | TargetSystem | Nested target config. |
| | dft | DFTConfig | Nested DFT config. |
| | runtime | RuntimeConfig | Runtime paths. |

## 3. Design Architecture

### Configuration (Pydantic Models)

1.  **`TargetSystem`** (in `config/schemas/common.py`):
    -   `elements`: `List[str]`. Validator: Check against periodic table.
    -   `composition`: `Dict[str, float]`. Validator: Sum must be close to 1.0.
    -   `crystal_structure`: `Optional[str]`.

2.  **`DFTConfig`** (in `config/schemas/dft.py`):
    -   `pseudopotential_dir`: `Path`. Validator: Must be an existing directory.
    -   `ecutwfc`: `float`. Constraint: `gt=0`.
    -   `kspacing`: `float`. Constraint: `gt=0`.
    -   `nspin`: `int` (1 or 2).

3.  **`RuntimeConfig`** (in `config/models.py`):
    -   `database_path`: `Path`. Default: `mlip.db`.
    -   `work_dir`: `Path`. Default: `_work`.

4.  **`MLIPConfig`** (in `config/models.py`):
    -   Aggregates `TargetSystem`, `DFTConfig`, `RuntimeConfig`.

### Database Layer (`DatabaseManager`)
The `DatabaseManager` class abstracts the `ase.db` library.

-   **Schema Enforcement**:
    -   `status`: Enum (`pending`, `running`, `completed`, `failed`).
    -   `config_type`: String (`sqs`, `md_snapshot`, `dimer`).
    -   `generation`: Integer.

-   **Methods**:
    -   `initialize()`: Creates the `.db` file.
    -   `add_structure(atoms: Atoms, metadata: dict)`: Inserts an atom.
    -   `count(filters: dict)`: Wraps `db.count()`.
    -   `update_status(id: int, status: str)`: Atomic update.

### CLI (`app.py`)
-   **Command**: `mlip-auto init`
-   **Command**: `mlip-auto check-config <file>`
-   **Command**: `mlip-auto db init`

## 4. Implementation Approach

1.  **Define Domain Primitives**: `config/schemas/common.py`.
2.  **Build Configuration Schemas**: `dft.py`, `models.py`.
3.  **Implement Logging**: `core/logging.py` using `Rich`.
4.  **Develop Database Manager**: `core/database.py`.
5.  **Create Service Layer**: `core/services.py` with `load_config`.
6.  **Construct CLI**: `app.py` using Typer.

## 5. Test Strategy
-   **Unit Testing**: `tests/unit/test_config.py`, `tests/unit/test_database.py`.
-   **Integration Testing**: `tests/e2e/test_cli.py`.

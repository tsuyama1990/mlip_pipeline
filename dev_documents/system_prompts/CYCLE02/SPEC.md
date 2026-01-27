# Cycle 02 Specification: Structure Generation & Database

## 1. Summary
Cycle 02 builds the data backbone of the system. We implement the **Structure Generator** to create initial training data and the **Database Manager** to store these structures efficiently. Additionally, we integrate the **Trainer** module, acting as a wrapper around the Pacemaker engine to manage datasets and potential files.

## 2. System Architecture

### 2.1. File Structure
```text
src/mlip_autopipec/
├── generator/                      # [CREATE]
│   ├── __init__.py
│   ├── random_builder.py           # [CREATE] Random structure generation
│   └── defects.py                  # [CREATE] Defect introduction logic
├── training/                       # [CREATE]
│   ├── __init__.py
│   ├── database.py                 # [CREATE] pckl.gzip handling
│   └── pacemaker_wrapper.py        # [CREATE] pace_train wrapper
└── orchestration/
    └── state.py                    # [MODIFY] Add dataset tracking
```

### 2.2. Component Interaction
- **`StructureGenerator`**: Called by Orchestrator to populate the initial pool. It uses `ASE` and `spglib` (optional) to generate diverse structures (distorted unit cells, vacancies, interstitials).
- **`DatabaseManager`**: Abstraction layer for Pacemaker's dataset format (`.pckl.gzip`). It ensures atomic structures are correctly converted to the format Pacemaker expects.
- **`PacemakerWrapper`**: Executes `pace_train` CLI commands via `subprocess`. It handles the `pair_style` logic (Delta learning setup).

## 3. Design Architecture

### 3.1. Database Manager (`src/mlip_autopipec/training/database.py`)
- **Responsibility**: Convert list of ASE `Atoms` into a Pandas DataFrame pickle (`.pckl.gzip`) compatible with Pacemaker.
- **Key Constraints**:
    - Must preserve `energy`, `force`, and `virial` (stress) tags.
    - Must validate that no `NaN` values exist in forces before saving.

### 3.2. Structure Generator (`src/mlip_autopipec/generator/`)
- **Strategies**:
    - **Rattled Bulk**: Apply random strain ($\pm 10\%$) and atomic displacement ($0.1 \AA$) to the primitive cell.
    - **Defects**: Remove random atoms (vacancies) or insert atoms (interstitials).
    - **Surfaces**: (Optional in Cycle 02) Create slab models.

### 3.3. Trainer (`src/mlip_autopipec/training/pacemaker_wrapper.py`)
- **Method**: `train(dataset_path: Path, initial_potential: Path = None) -> Path`
- **Logic**:
    - Constructs the `input.yaml` for Pacemaker.
    - Sets `fitting.weight_energy` and `fitting.weight_forces`.
    - Runs `pace_train`.
    - Returns path to the new `potential.yace`.

## 4. Implementation Approach

1.  **Database**: Implement `save_dataset(atoms_list, path)` and `load_dataset(path)`. Test round-trip fidelity.
2.  **Generator**: Implement `RandomBuilder` class. Input: `primitive_cell`, Output: `List[Atoms]`. Use ASE's `rattle` and `strain` methods.
3.  **Trainer**: Implement the `subprocess` call to `pace_train`. Ensure the environment (ACE) is accessible.

## 5. Test Strategy

### 5.1. Unit Testing
- **Database**: Create a list of dummy Atoms. Save to file. Load back. Assert positions and forces match exactly.
- **Generator**: Verify that generated structures have different cell parameters than the input (proof of distortion). Check that atom counts match expected (e.g., vacancy creation reduces count by 1).

### 5.2. Integration Testing
- **Trainer**: Mock the `subprocess.run` to avoid actually running heavy training. Verify that the correct arguments (`--dataset`, `--output-dir`) are passed to the command.

# Cycle 02: Structure Generation & Database Management

## 1. Summary

In Cycle 02, we implement the "Explorer" and the "Memory" of the system. We introduce the `StructureGenerator` module, which is responsible for creating the initial pool of atomic structures to start the learning process. This involves simple random perturbations and cell distortions (Rattling). Simultaneously, we implement the `DatabaseManager`, which handles the persistence of these structures and their associated DFT results. Finally, we introduce the `Trainer` module (basic version), which wraps the `pacemaker` executable to enable the training of the first generation of ACE potentials from the database.

By the end of this cycle, the system will be able to: 1) Generate random structures, 2) Save them to a persistent database, 3) (Conceptually) run DFT on them (using Cycle 01's Oracle), and 4) Train a preliminary potential.

## 2. System Architecture

### File Structure

**mlip_autopipec/**
├── **database/**
│   ├── **__init__.py**
│   └── **manager.py**          # DatabaseManager class
├── **generator/**
│   ├── **__init__.py**
│   ├── **explorer.py**         # InitialExplorer class
│   └── **defects.py**          # Defect generation logic
└── **trainer/**
    ├── **__init__.py**
    └── **pace.py**             # PacemakerRunner class

### Component Description

*   **`generator/explorer.py`**: Implements the logic to create diverse structures. For the "Cold Start", it supports:
    *   **Rattling**: Random displacement of atoms (Gaussian noise).
    *   **Scaling**: Isotropic and anisotropic volume changes to sample the Equation of State (EOS).
*   **`database/manager.py`**: Manages the storage of training data. It wraps `ase.io` to save/load structures in compressed formats (e.g., `.pckl.gzip` for Pacemaker compatibility, or `.xyz` for portability). It also handles metadata (e.g., "This structure came from Generation 1, Random Exploration").
*   **`trainer/pace.py`**: A wrapper for the `pace_train` command. It takes a dataset path and a config, generates the `input.yaml` for Pacemaker, and executes the training process.

## 3. Design Architecture

### Domain Models

**`StructureMetadata`**
*   **Role**: provenance tracking.
*   **Fields**:
    *   `source`: `str` (e.g., "initial_random", "md_halt", "active_learning_iter_3").
    *   `generation`: `int`.
    *   `labels`: `List[str]` (e.g., ["train", "validation"]).

**`TrainingConfig`** (in `config/schemas/training.py`)
*   **Role**: Configuration for the Pacemaker training.
*   **Fields**:
    *   `batch_size`: `int`
    *   `max_num_epochs`: `int`
    *   `energy_weight`: `float`
    *   `force_weight`: `float`
    *   `cutoff`: `float` (RCut)
    *   `ladder_step`: `List[int]` (Basis set size definition)

### Key Invariants
1.  **Data Integrity**: The database must never store structures with NaN positions or undefined cells.
2.  **Format Compatibility**: The Trainer must export the ASE atoms into the specific format required by Pacemaker (`.pckl.gzip` or specific ExtXYZ format) before invoking the executable.
3.  **Reproducibility**: Random number generators in the Explorer must be seedable via the configuration to ensure reproducible structure generation.

## 4. Implementation Approach

1.  **Database Manager**:
    *   Implement `add_structure(atoms, metadata)`.
    *   Implement `save_dataset(filepath)`: Aggregates all structures and saves them to a single file for training.
    *   Implement `load_dataset(filepath)`: Loads structures back.

2.  **Structure Generator**:
    *   Implement `InitialExplorer.generate(input_structure, count)`.
    *   Apply random strain (using `numpy`) to the cell.
    *   Apply random displacement to positions.

3.  **Trainer Wrapper**:
    *   Implement `PaceRunner.train(dataset_path, output_dir)`.
    *   Construct the `input.yaml` file dynamically based on `TrainingConfig`.
    *   Execute `pace_train input.yaml`.
    *   Parse the output to find the path to the final `potential.yace` file.

## 5. Test Strategy

### Unit Testing
*   **Generator**: Verify that `generate()` produces the requested number of structures and that they are indeed different from the input (positions have changed). Check that cell volume changes are within the specified bounds.
*   **Database**: Create a list of Atoms, save them, load them back, and assert equality (positions, numbers, cell).
*   **Trainer Config**: Verify that `TrainingConfig` generates a valid YAML string that matches Pacemaker's expected input format.

### Integration Testing
*   **Generator -> Database**: Generate 10 structures, add them to the DB, and verify the DB count is 10.
*   **Trainer (Mocked)**: Mock the `subprocess.run` call in `PaceRunner`. Verify that the correct command line arguments are passed (`pace_train input.yaml`) and that the `input.yaml` file is created in the temp directory.

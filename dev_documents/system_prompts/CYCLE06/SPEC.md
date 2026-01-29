# Cycle 06 Specification: Active Learning Loop

## 1. Summary

Cycle 06 enables the **Dynamics Engine** to perform "On-the-Fly" (OTF) active learning. This is the core engine of the system. It implements the interface to LAMMPS, enabling Molecular Dynamics (MD) simulations that are automatically interrupted when the extrapolation grade ($\gamma$) exceeds a threshold. When halted, the system extracts the problematic structure, cuts out a cluster, embeds it in a vacuum-padded supercell, and sends it to the Oracle for labeling. This closes the active learning loop.

## 2. System Architecture

### File Structure

Files to be created/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── config.py                     # Update: Add DynamicsConfig (LAMMPS)
├── **modules/**
│   └── **cli_handlers/**
│       └── **handlers.py**           # Update: Add run-loop logic
├── **inference/**
│   ├── **__init__.py**
│   └── **lammps.py**                 # LAMMPS wrapper with fix halt
└── orchestration/
    ├── **candidate_processing.py**   # Logic to extract/embed structures
    └── phases/
        ├── **__init__.py**
        └── **dynamics.py**           # DynamicsPhase implementation
```

## 3. Design Architecture

### Domain Models

#### `config.py`
*   **`DynamicsConfig`**:
    *   `md_steps`: int
    *   `timestep`: float
    *   `temperature`: float
    *   `uncertainty_threshold`: float (gamma)
    *   `check_interval`: int (steps between gamma checks)

### Components (`inference/lammps.py`)

#### `LammpsRunner`
*   **`run_md(structure, potential, config) -> RunResult`**:
    *   Generates `in.lammps`.
    *   **Crucial**: Sets up `pair_style hybrid/overlay pace zbl`.
    *   Sets up `compute pace` and `fix halt` to stop if gamma > threshold.
    *   Returns status (`COMPLETED` or `HALTED`) and path to dump file.

### Orchestration (`orchestration/candidate_processing.py`)

#### `CandidateManager`
*   **`extract_halt_structure(dump_file) -> Structure`**: Reads the last frame.
*   **`cut_cluster(structure, center_atom, radius) -> Structure`**: Cuts a cluster.
*   **`periodic_embed(cluster, buffer) -> Structure`**: Places cluster in a box for DFT.

### Orchestration (`orchestration/phases/dynamics.py`)

#### `DynamicsPhase`
*   Runs MD using `LammpsRunner`.
*   If halted:
    *   Calls `CandidateManager` to process the structure.
    *   Adds new candidates to `WorkflowState`.
    *   Triggers transition back to ORACLE phase.
*   If completed:
    *   Transition to DONE (or next iteration).

## 4. Implementation Approach

1.  **Update Config**: Add `DynamicsConfig`.
2.  **Implement LAMMPS Runner**:
    *   Use `lammps` Python module or `subprocess`.
    *   Ensure `hybrid/overlay` is correctly scripted.
    *   Implement `fix halt` logic.
3.  **Implement Candidate Processing**:
    *   Implement logic to find the atom with max gamma.
    *   Implement cluster cutting and embedding (crucial for valid DFT).
4.  **Implement Dynamics Phase**:
    *   Wire the loop: MD -> Halt -> Extract -> Oracle.

## 5. Test Strategy

### Unit Testing
*   **`test_candidate_processing.py`**:
    *   Create a structure. Cut a cluster. Verify the returned structure has enough vacuum padding.
*   **`test_lammps_runner.py`**:
    *   Verify `in.lammps` generation contains `fix halt` and `pair_style hybrid`.

### Integration Testing
*   **`test_dynamics_phase.py`**:
    *   Mock LAMMPS execution.
    *   Simulate a "Halt" return status.
    *   Verify that new candidates are added to the workflow state and the phase transitions to Oracle.

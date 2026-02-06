# Cycle 05 Specification: Dynamics Engine Basic (MD & Inference)

## 1. Summary

Cycle 05 activates the **Dynamics Engine**. Now that we have a potential (Cycle 04), we need to run Molecular Dynamics (MD) simulations to explore the phase space. This module acts as the "Explorer" in the active learning loop, but unlike the random `StructureGenerator`, it follows the physical trajectory defined by the learned potential.

We will use **LAMMPS** as the backend. The critical feature here is the **Hybrid Potential Overlay**. We must ensure that the MD simulation uses the *sum* of the ACE potential and the ZBL baseline, matching exactly how the potential was trained (Delta Learning). Failure to do this will result in inaccurate forces or simulation crashes.

## 2. System Architecture

Files to create/modify are **bolded**.

```ascii
src/mlip_autopipec/
├── config/
│   └── config_model.py         # Update ExplorerConfig (add MD params)
└── infrastructure/
    └── dynamics/
        ├── **__init__.py**
        └── **lammps_adapter.py** # Wrapper for LAMMPS
```

## 3. Design Architecture

### 3.1. Dynamics Configuration
*   We treat `LammpsDynamics` as a type of `Explorer`.
*   Config: `temperature`, `pressure`, `n_steps`, `timestep`.

### 3.2. LammpsAdapter Logic
*   **Input Generation**: Writes `in.lammps` dynamically.
    *   **CRITICAL**: Must write `pair_style hybrid/overlay pace zbl ...`.
    *   Must write `pair_coeff` for both PACE and ZBL.
*   **Execution**: Can use the `lammps` Python binding (preferred) or `subprocess` call to `lmp_serial/lmp_mpi`.
*   **Output**: Reads `dump.lammps` to retrieve the final structure and trajectory.

## 4. Implementation Approach

1.  **Template Engine**: Create a robust template for `in.lammps`.
2.  **Hybrid Logic**: Implement the logic that maps species (e.g., Fe, Pt) to their ZBL atomic numbers and interaction radii.
3.  **Runner**: Implement `LammpsRunner` class.
    *   `run_md(structure: Atoms, potential_path: Path, settings: Dict) -> ExplorationResult`
4.  **Parser**: Use `ase.io.read` to parse the LAMMPS dump file.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Input Check**: Generate an input file for Fe-Pt.
    *   Assert `pair_style hybrid/overlay` is present.
    *   Assert `pair_coeff * * zbl` is present.
    *   Assert `pair_coeff * * pace` is present.

### 5.2. Integration Testing (Mocked LAMMPS)
*   **Mock**: Create a mock `lammps` object or executable.
*   **Run**: Call `LammpsAdapter.run_md`.
*   **Verify**: Ensure the code correctly identifies the potential file path and passes it to the runner. Ensure it parses the (dummy) dump file correctly.

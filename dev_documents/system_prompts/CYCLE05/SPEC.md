# Cycle 05 Specification: Dynamics Engine (The Executor - MD)

## 1. Summary

In this cycle, we implement the **Dynamics Engine** module, focusing on the integration of **LAMMPS** for Molecular Dynamics (MD) simulations. This engine is responsible for running the simulations that explore the configuration space, testing the potential generated in the previous cycles.

A critical requirement is the robust handling of **Hybrid Potentials**. Since our Trainer learns a residual (Delta Learning), the Dynamics Engine must automatically construct the appropriate LAMMPS input commands to overlay the ML potential (ACE) with the physics baseline (ZBL/LJ). This ensures that the simulation remains stable even during high-energy collisions, preventing the "exploding atoms" problem common in pure ML potentials.

We will focus on "standard" MD execution here (NVE, NVT, NPT). The active learning "Halt" logic will be added in Cycle 06.

## 2. System Architecture

The following file structure will be created/modified. Files in **bold** are the specific deliverables for this cycle.

```ascii
src/mlip_pipeline/
├── components/
│   ├── dynamics/
│   │   ├── **__init__.py**
│   │   ├── **lammps_driver.py**   # Low-level LAMMPS I/O
│   │   └── **hybrid.py**          # Hybrid Potential Logic
│   └── base.py                    # (Modified) Enhance BaseDynamics interface
```

## 3. Design Architecture

### 3.1. Dynamics Interface
The `BaseDynamics` in `src/mlip_pipeline/components/base.py` will be refined.

*   `explore(self, initial_structure: Structure, potential: Potential, settings: Dict) -> Trajectory`
    *   Input: Starting structure, the potential to use, and simulation settings (Temp, steps).
    *   Output: A `Trajectory` object (or path to dump file) containing the simulation history.

### 3.2. LAMMPS Driver
Located in `src/mlip_pipeline/components/dynamics/lammps_driver.py`.

*   **Config**: `LAMMPSDynamicsConfig`.
    *   `command`: String (e.g., `lmp_serial`).
    *   `n_cores`: Int.
*   **Logic**:
    1.  **Input Generation**: Create `data.lammps` (structure) and `in.lammps` (control).
    2.  **Hybrid Setup**: Call `hybrid.generate_pair_style` to create the complex `pair_style hybrid/overlay` command.
    3.  **Execution**: Run LAMMPS.
    4.  **Parsing**: Convert `dump.lammps` back to a list of `Structure` objects (if needed) or just return the path.

### 3.3. Hybrid Potential Logic
Located in `src/mlip_pipeline/components/dynamics/hybrid.py`.
*   **Function**: `generate_pair_style(potential: Potential, elements: List[str]) -> str`
*   **Logic**:
    *   Check `potential.baseline_type`.
    *   If "ZBL":
        *   Generate `pair_style hybrid/overlay pace zbl 1.0 2.0`.
        *   Generate `pair_coeff * * pace potential.yace Element1 Element2`.
        *   Generate `pair_coeff * * zbl Z1 Z2`.
    *   If "LJ":
        *   Similar logic but with `lj/cut`.

## 4. Implementation Approach

1.  **Hybrid Utility**: Implement `hybrid.py` first. This is purely string manipulation based on element types and potential metadata.
2.  **LAMMPS IO**: Use `ase.io.lammpsdata.write_lammps_data` to write the structure file.
3.  **Input Template**: Create a flexible Jinja2 template or string builder for `in.lammps` that accepts thermostat settings (fix nvt), timestep, and the generated pair styles.
4.  **Driver**: Implement the class that orchestrates the file writing, execution, and cleanup.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Hybrid String Generation**: Pass a potential with ZBL baseline and elements [Fe, Pt]. Assert the returned string contains correct `pair_style hybrid/overlay` and `pair_coeff` lines with correct atomic numbers.
*   **Input File Generation**: Generate an input file for NVT at 300K. Check for `fix 1 all nvt temp 300.0 300.0 ...`.

### 5.2. Integration Testing
*   **Mock Execution**: Run a short MD (10 steps) using a mock potential (or LJ if PACE is not installed). Assert that `log.lammps` and `dump.lammps` are created.
*   **Energy Conservation**: Run NVE simulation. Assert total energy drift is minimal (< 1 meV/atom/ps).
*   **Core Repulsion Check**: Initialize two atoms very close (0.5 Å). Run 1 step. Assert they fly apart (strong repulsive force) and don't overlap further.

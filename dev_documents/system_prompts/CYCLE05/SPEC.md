# Cycle 05 Specification: Dynamics Engine (Basic MD)

## 1. Summary

Cycle 05 establishes the "User" component: the Dynamics Engine. Once a potential is trained, it must be used to simulate physical phenomena. This cycle focuses on implementing the interface to **LAMMPS**, the standard engine for Molecular Dynamics (MD).

Critically, this cycle implements the **Hybrid Potential** safety mechanism defined in the architecture. Instead of running a pure ACE potential (which might be unstable), we construct a `hybrid/overlay` potential that sums the ACE forces with a Physics Baseline (ZBL or LJ). This ensures that even if the ML model predicts zero repulsion at short distances, the physics baseline prevents atomic overlap and simulation crashes.

By the end of this cycle, the Orchestrator will be able to run stable MD simulations (NPT/NVT) using the trained potential.

## 2. System Architecture

This cycle focuses on the `components/dynamics` package.

### File Structure
Files to be created/modified in this cycle are marked in **bold**.

```
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── **base.py**             # Enhanced Abstract Base Class
│   │   ├── **lammps_driver.py**    # Interface to LAMMPS
│   │   └── **hybrid.py**           # Logic to generate pair_style commands
│   └── factory.py                  # Update to register Dynamics
├── domain_models/
│   └── **config.py**               # Add DynamicsConfig details
└── tests/
    └── **test_dynamics.py**
```

## 3. Design Architecture

### 3.1. Dynamics Configuration (`domain_models/config.py`)
Update `DynamicsConfig` to include:
*   `type`: "lammps".
*   `ensemble`: "npt" or "nvt".
*   `temperature`: float (K).
*   `pressure`: float (bar).
*   `timestep`: float (fs, e.g., 1.0).
*   `n_steps`: int.
*   `uncertainty_threshold`: float (for Cycle 06, but defined here).

### 3.2. Hybrid Potential Generator (`components/dynamics/hybrid.py`)
This module generates the LAMMPS commands to overlay potentials.
*   **Input**: `Potential` object (path to .yace, baseline config).
*   **Output**: List of strings (LAMMPS commands).
*   **Example Output**:
    ```
    pair_style hybrid/overlay pace zbl 0.5 2.0
    pair_coeff * * pace potential.yace Li O
    pair_coeff * * zbl 3 8
    ```

### 3.3. LAMMPS Driver (`components/dynamics/lammps_driver.py`)
We wrap `lammps` (using the Python interface or subprocess).
*   `run_exploration(structure, potential, config)`:
    1.  Write `data.lammps`.
    2.  Write `in.lammps`:
        *   Setup units (metal).
        *   Setup hybrid pair style.
        *   Setup `fix npt`.
        *   Setup `dump`.
    3.  Execute LAMMPS.
    4.  Return final structure and trajectory path.

## 4. Implementation Approach

1.  **Implement `HybridPotentialGenerator`**: Create the logic to map element symbols to ZBL atomic numbers and generate the correct `pair_coeff` lines.
2.  **Implement `LAMMPSDynamics`**:
    *   Use `ase.io.lammpsdata.write_lammps_data`.
    *   Construct the input script using a template system (Jinja2 or f-strings).
    *   Ensure the `pace` pair style is available (requires a LAMMPS build with `USER-PACE`).
3.  **Mocking**: Since `USER-PACE` might not be in the standard CI container, create a `MockLAMMPS` that pretends to run MD and returns a perturbed structure.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Hybrid Command Generation**:
    *   Input: `Potential` with ZBL baseline, Elements ["Fe", "Pt"].
    *   Action: `generate_commands()`.
    *   Assert: Output contains `pair_style hybrid/overlay` and `pair_coeff * * zbl 26 78`.

### 5.2. Integration Testing
*   **LAMMPS Execution (Mocked)**:
    *   Input: Structure, Potential.
    *   Action: `dynamics.explore()`.
    *   Assert: `in.lammps` file is created in the work directory.
    *   Assert: `dump.lammps` is "created" (by the mock).
    *   Assert: Returns a valid `Structure` object representing the final state.

# Cycle 05: Dynamics (LAMMPS) Specification

## 1. Summary

Cycle 05 focuses on implementing the **Dynamics Engine** module using **LAMMPS**. This component is critical for the active learning loop, as it explores the potential energy surface (PES) to find configurations where the current potential is uncertain. A key feature is the **Hybrid Potential** mechanism, which overlays the machine-learned ACE potential with a physics-based baseline (Lennard-Jones or ZBL). This ensures that the simulation remains stable even when exploring high-energy regions where the ML potential might be poorly defined, preventing "holes" in the PES and catastrophic failures like atomic fusion.

## 2. System Architecture

This cycle focuses on the `components/dynamics` package and integration with `lammps` (via Python interface or file-based execution).

### File Structure

The following file structure will be created. **Bold** files are to be implemented in this cycle.

*   **`src/`**
    *   **`mlip_autopipec/`**
        *   **`components/`**
            *   **`dynamics/`**
                *   **`__init__.py`**
                *   **`base_dynamics.py`** (Abstract Base Class)
                *   **`lammps_dynamics.py`** (Main Implementation)
                *   **`lammps_driver.py`** (Input Script Generator)
                *   **`hybrid.py`** (Hybrid Potential Logic)

## 3. Design Architecture

### 3.1 Components

#### `BaseDynamics`
Defines the standard interface for MD engines.
*   **`explore(potential: PotentialArtifact, context: dict) -> ExplorationResult`**:
    *   Input: A trained potential and context (temperature, pressure, steps).
    *   Output: An `ExplorationResult` containing the final structure, trajectory file path, and whether the simulation halted early due to uncertainty.

#### `LAMMPSDynamics`
Concrete implementation for LAMMPS.
*   **`__init__(config: LAMMPSDynamicsConfig)`**: Sets up MD parameters (timestep, thermostat).
*   **`_run_md(potential: PotentialArtifact) -> ExplorationResult`**: Executes the LAMMPS simulation. Handles file setup and cleanup.
*   **`_check_halt(log_file: Path) -> bool`**: Parses the LAMMPS log to determine if `fix halt` was triggered (will be fully implemented in Cycle 06, stub here).

#### `LAMMPSDriver`
Responsible for generating the `in.lammps` script.
*   **`generate_input(structure: Structure, potential: PotentialArtifact, config: LAMMPSDynamicsConfig) -> str`**:
    *   Generates commands for:
        *   Initialization (`units metal`, `boundary p p p`).
        *   Structure definition (`read_data`).
        *   Potential setup (delegates to `hybrid.py`).
        *   MD settings (`fix npt`, `timestep`).
        *   Output (`dump`, `thermo`).

#### `hybrid.py`
Helper for hybrid potential generation.
*   **`generate_pair_style(elements: list[str], potential_path: Path) -> list[str]`**:
    *   Returns the `pair_style hybrid/overlay` command and `pair_coeff` lines.
    *   Example:
        ```lammps
        pair_style hybrid/overlay pace zbl 1.0 2.0
        pair_coeff * * pace potential.yace Mg O
        pair_coeff * * zbl 12 8
        ```

### 3.2 Domain Models

*   **`ExplorationResult`**:
    *   `final_structure: Structure`
    *   `trajectory_path: Path`
    *   `halted: bool`
    *   `meta: dict` (e.g., `steps_completed`)

*   **`LAMMPSDynamicsConfig`**:
    *   `timestep: float` (e.g., 0.001 ps)
    *   `temperature: float`
    *   `pressure: float`
    *   `steps: int`
    *   `thermostat: str` (e.g., "nose-hoover")

## 4. Implementation Approach

1.  **Hybrid Logic**: Implement `hybrid.py` first. Ensure it correctly maps element symbols to atomic numbers for ZBL/LJ.
2.  **Driver**: Implement `LAMMPSDriver`. Use a template-based approach or string builder for the input script. Ensure paths are absolute.
3.  **ASE Integration**: Use `ase.io.write` to generate the `data.lammps` file from the input `Structure`.
4.  **Execution**: Implement `LAMMPSDynamics._run_md`. Use `subprocess` to call `lmp_serial` or `lmp_mpi`. Alternatively, use the `lammps` Python module if available (preferred for tighter integration).
5.  **Configuration**: Update `config.py` with `DynamicsConfig` and `LAMMPSDynamicsConfig`.
6.  **Factory**: Register `LAMMPSDynamics` in `ComponentFactory`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **`test_hybrid.py`**:
    *   Input: `["Fe", "Pt"]`.
    *   Output: Verify `pair_coeff * * zbl 26 78` is generated.
    *   Verify `pair_style hybrid/overlay` is present.
*   **`test_driver.py`**:
    *   Initialize `LAMMPSDriver` with dummy config.
    *   Generate input script.
    *   Verify essential commands (`fix npt`, `dump`) are present.
    *   Verify that `potential.yace` path is correctly quoted.

### 5.2 Integration Testing (Mocked/CI)
*   **Mock Execution**:
    *   Mock `subprocess.run` or `lammps` object.
    *   Call `dynamics.explore(potential)`.
    *   Verify that `data.lammps` and `in.lammps` are created in the work directory.
    *   Verify that cleanup logic removes temporary files (optional).

### 5.3 Integration Testing (Real - Optional)
*   **Real MD Run**:
    *   Requires `lmp` executable.
    *   Run a short NVT simulation (100 steps) of Argon (LJ).
    *   Verify that the temperature fluctuates around the target.
    *   Verify that `ExplorationResult` contains a valid `final_structure`.

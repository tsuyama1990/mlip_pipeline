# Cycle 07: Advanced Dynamics (EON/kMC) Specification

## 1. Summary

Cycle 07 expands the simulation capabilities of PyAceMaker beyond the nanosecond timescale of Molecular Dynamics (MD) to the seconds, hours, or even years timescale of diffusive processes. This is achieved by integrating **Adaptive Kinetic Monte Carlo (aKMC)** via the **EON** software package. The primary goal is to allow the system to seamlessly transition from dynamic deposition (MD) to long-term structural ordering (kMC). A critical component is the development of a custom Python driver (`pace_driver.py`) that enables EON to use the Pacemaker potential for energy and force calculations, including on-the-fly uncertainty monitoring during saddle point searches.

## 2. System Architecture

This cycle focuses on the `components/dynamics/eon_driver.py` module and the external `pace_driver.py` script.

### File Structure

The following file structure will be created. **Bold** files are to be implemented in this cycle.

*   **`src/`**
    *   **`mlip_autopipec/`**
        *   **`components/`**
            *   **`dynamics/`**
                *   **`eon_driver.py`** (Main EON Wrapper)
                *   **`templates/`**
                    *   **`pace_driver.py`** (EON-Pacemaker Interface Script)
                    *   **`config.ini`** (EON Configuration Template)

## 3. Design Architecture

### 3.1 Components

#### `EONDriver`
Manages the execution of EON simulations.
*   **`__init__(config: EONDynamicsConfig)`**: Sets up kMC parameters (temperature, prefactor, search method).
*   **`setup_simulation(structure: Structure, potential: PotentialArtifact) -> Path`**:
    *   Creates the EON working directory.
    *   Writes `reactant.con` (initial structure).
    *   Generates `config.ini` based on settings.
    *   Deploys `pace_driver.py` and the potential file.
*   **`run_kmc() -> ExplorationResult`**:
    *   Executes the EON client (`eonclient`).
    *   Monitors for special exit codes (e.g., 100) indicating high uncertainty.
    *   Parses EON output (`processes.dat`, `energy.dat`) to track time and events.

#### `pace_driver.py` (Template)
A standalone script executed by EON.
*   **`get_energy_forces(coords) -> (float, list[float])`**:
    *   Input: Atomic coordinates from EON (via stdin or file).
    *   Output: Potential energy and forces (via stdout).
    *   Logic:
        1.  Reconstruct `ase.Atoms` object.
        2.  Calculate energy/forces using `pacemaker.calculator`.
        3.  **Check $\gamma$**: Calculate max extrapolation grade.
        4.  **Halt**: If $\gamma > \text{threshold}$, write the structure to `bad_structure.con` and exit with code 100.

### 3.2 Domain Models

*   **`EONDynamicsConfig`**:
    *   `temperature: float`
    *   `superbasin_scheme: bool`
    *   `saddle_search_method: str` (e.g., "dimer")
    *   `confidence: float` (for confidence limit termination)

## 4. Implementation Approach

1.  **Template Generation**: Create the `pace_driver.py` template. This script must be robust and self-contained, as it runs outside the main Python process.
2.  **Configuration**: Implement `EONDynamicsConfig` in `config.py`.
3.  **Input Generation**: Implement `EONDriver.setup_simulation`. Ensure `reactant.con` format is correct (EON uses a specific format).
4.  **Execution Wrapper**: Implement `run_kmc`. Use `subprocess.Popen` to capture stdout/stderr in real-time.
5.  **Halt Handling**: Implement logic to detect the special exit code (100) from the driver script and treat it as a "Halt" event in the Orchestrator.
6.  **Orchestrator Update**: Allow the Orchestrator to switch between `LAMMPSDynamics` and `EONDynamics` based on the cycle stage or configuration.

## 5. Test Strategy

### 5.1 Unit Testing
*   **`test_eon_config.py`**:
    *   Initialize `EONDriver` with specific settings.
    *   Generate `config.ini`.
    *   Verify parameters (e.g., `temperature = 600`).
*   **`test_pace_driver.py` (Standalone)**:
    *   Run the driver script manually with a dummy potential and structure input.
    *   Verify it prints energy and forces in the correct format.
    *   Verify it exits with code 100 when passed a high-gamma structure (mocked).

### 5.2 Integration Testing (Mocked EON)
*   **Mock Execution**:
    *   Mock `subprocess.run` to simulate `eonclient` running for a few steps.
    *   Call `driver.run_kmc()`.
    *   Verify that `ExplorationResult` contains the accumulated time.

### 5.3 Integration Testing (Real - Optional)
*   **Real kMC Run**:
    *   Requires `eonclient` installed.
    *   Run a short kMC simulation on an adatom diffusion process (e.g., Al on Al(100)).
    *   Verify that it finds saddle points and advances the system time.
    *   Verify that the "Halt" mechanism triggers if the potential is poor.

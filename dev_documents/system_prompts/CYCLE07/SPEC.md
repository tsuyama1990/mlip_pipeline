# Cycle 07 Specification: Dynamics Engine II (EON & aKMC)

## 1. Summary
This cycle extends the temporal reach of the system by integrating Adaptive Kinetic Monte Carlo (aKMC) via the EON software suite. While MD (Cycle 05) is limited to nanoseconds, aKMC can simulate seconds or hours of diffusion and ordering events. The core challenge here is interoperability: EON is a C++ code that typically expects internal potentials. We must implement a "Bridge" that allows EON to call our Python-based ACE potential for energy and force calculations. Furthermore, we must replicate the "Uncertainty Watchdog" within this bridge: if EON explores a transition state (saddle point) that has high uncertainty, the bridge must signal EON to abort, triggering the same OTF learning loop as in Cycle 06.

## 2. System Architecture

### 2.1. File Structure
files to be created/modified in this cycle are bolded.

```text
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── eon_driver.py           # [CREATE] EON Wrapper
│   │   └── eon_bridge.py           # [CREATE] The script called by EON
│   └── utils/
│       └── file_formats.py         # [MODIFY] Add .con (EON) format support
├── domain_models/
│   └── config.py                   # [MODIFY] Add EONConfig
```

### 2.2. Component Interaction
1.  **`Orchestrator`** calls `dynamics.run_akmc(potential, context=state)`.
2.  **`EONDriver`**:
    *   Creates a directory `kmc_run/`.
    *   Writes `config.ini` for EON, setting `potential = script` and `script_path = .../eon_bridge.py`.
    *   Converts current structure to `reactant.con`.
    *   Executes `eonclient` (or similar executable).
3.  **`eonclient`** (External Process):
    *   Calls `python eon_bridge.py` repeatedly for energy/forces.
4.  **`eon_bridge.py`**:
    *   Loads `potential.yace`.
    *   Reads coordinates from stdin (EON protocol).
    *   Calculates Energy, Forces, and **Gamma**.
    *   **Check**: If $\gamma > threshold$:
        *   Writes the bad structure to `bad.con`.
        *   Exits with a specific error code (e.g., 100).
    *   Else:
        *   Prints Energy and Forces to stdout.
5.  **`EONDriver`**:
    *   Catches exit code 100 -> Returns `HALTED`.
    *   Catches normal completion -> Returns `CONVERGED`.

## 3. Design Architecture

### 3.1. Domain Models

#### `config.py`
*   `EONConfig`:
    *   `temperature`: float
    *   `process_search_method`: Literal['dimer', 'lanczos']
    *   `eon_executable_path`: str

### 3.2. Core Logic

#### `eon_bridge.py` (The Interceptor)
*   **Responsibility**: Serve physics to EON while acting as a policeman.
*   **Protocol**: EON Potentials Interface (usually text-based stdin/stdout).
    *   Input: Number of atoms, Lattice vectors, Coordinates.
    *   Output: Energy, Forces (Nx3).
*   **Performance**: Loading the ACE potential takes time. The bridge might need to be a persistent server or use fast loading mechanisms. For Cycle 07, standard loading is acceptable if slow.

#### `eon_driver.py`
*   **Responsibility**: Setup and teardown.
*   **File Formats**: EON uses `.con` files. We need a converter `ase_to_con` and `con_to_ase`.

## 4. Implementation Approach

### Step 1: File Conversion
*   Implement `ase_to_con` and `con_to_ase` in `utils/file_formats.py`.

### Step 2: The Bridge Script
*   Implement `src/mlip_autopipec/components/dynamics/eon_bridge.py`.
*   This script must be standalone executable. It will import `pacemaker` calculator.

### Step 3: EON Driver
*   Implement `EONDriver`.
*   Ensure it sets up the environment (PYTHONPATH) so the bridge can find the library.

### Step 4: Integration
*   Update `Orchestrator` to switch between MD and kMC based on config or policy.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_file_formats.py`**:
    *   Roundtrip conversion: Atoms -> .con -> Atoms. Check positions match.
*   **`test_eon_bridge.py`**:
    *   Pipe a dummy structure into the bridge script's stdin.
    *   Assert stdout contains numbers (Energy/Forces).
    *   Test "High Gamma" case: Mock the calculator to return high gamma, assert script exits with code 100.

### 5.2. Integration Testing
*   **`test_akmc_mock.py`**:
    *   Since EON binary might be missing, create a `MockEONClient` (python script) that calls the bridge script 10 times and then exits.
    *   Verify `EONDriver` handles the interaction.

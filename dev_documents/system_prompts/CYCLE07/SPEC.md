# Cycle 07 Specification: Advanced Dynamics (EON & Deposition)

## 1. Summary

In this cycle, we extend the system's capabilities to handle complex, long-timescale phenomena. We integrate the **EON** (Eon Client/Server) software for Adaptive Kinetic Monte Carlo (aKMC) simulations, allowing us to bridge the gap between nanoseconds (MD) and seconds (real-world aging/diffusion). Additionally, we implement a specialized **Deposition Module** in LAMMPS to simulate thin film growth (PVD), addressing the "Grand Challenge" of Fe/Pt ordering.

## 2. System Architecture

The following file structure will be created. **Bold** files are the targets for this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── **eon.py**              # Wrapper for EON Client
│   │   └── **deposition.py**       # LAMMPS Deposition Script Generator
│   └── **eon_driver.py**           # Python Driver for EON -> Pacemaker
tests/
└── **test_advanced_dynamics.py**   # Tests for EON Config & Deposition Logic
```

## 3. Design Architecture

### 3.1. EON Integration (`components/dynamics/eon.py`)
*   **`EONWrapper`**:
    *   `run_kmc(structure, potential) -> KMCResult`:
        1.  **Setup**: Create EON working directory (`config.ini`, `reactant.con`, `potentials/`).
        2.  **Driver**: Copy `eon_driver.py` into the `potentials/` folder. EON will call this script to get Energy/Forces from Pacemaker.
        3.  **Execute**: Run `eonclient` (or equivalent command) in a loop until a transition is found.
        4.  **Parse**: Read `process_search.log` or EON output to get the new structure and time elapsed.
        5.  **OTF Check**: The `eon_driver.py` itself contains the uncertainty check. If $\gamma > threshold$, it exits with a specific code, signaling a Halt to the Orchestrator.

### 3.2. EON Driver (`components/eon_driver.py`)
*   **`EONDriver`**: A standalone script (template) that:
    *   Reads atomic coordinates from EON (stdin or file).
    *   Loads the ACE potential using `ase.calculators.lammpsrun` (or pace calculator).
    *   Calculates Energy and Forces.
    *   Checks Uncertainty ($\gamma$).
    *   Writes Energy/Forces to stdout for EON.
    *   Exits with error if $\gamma$ is high.

### 3.3. Deposition Module (`components/dynamics/deposition.py`)
*   **`DepositionDynamics`**:
    *   `run_deposition(substrate, potential, species, rate, temp) -> DynamicsResult`:
        *   Generates a complex LAMMPS script using `fix deposit`.
        *   Configures `region` (where atoms appear) and `velocity` (thermal).
        *   Uses `fix nvt` or `fix nve` for the substrate.
        *   Uses `hybrid/overlay` (ACE+ZBL) to prevent crashes on impact.

## 4. Implementation Approach

1.  **EON Driver**: Write the `eon_driver.py` template. Ensure it works with the EON API (simple I/O).
2.  **EON Wrapper**: Implement `EONWrapper.run_kmc`. Handle config file generation (`config.ini`).
3.  **Deposition Logic**: Implement `DepositionDynamics`. Verify `fix deposit` syntax.
4.  **Integration**: Add `EONConfig` and `DepositionConfig` to `config.py`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_deposition.py`**:
    *   Verify generated LAMMPS script contains `fix deposit` with correct parameters (target region, species).

### 5.2. Integration Testing (Mocked Binary)
*   **`test_eon.py`**:
    *   **Mock `eonclient`**: Simulate a process search that finds a saddle point.
    *   **Verify Driver**: Run `eon_driver.py` directly with a test structure and potential. Assert it outputs energy/forces correctly.
    *   **Verify Halt**: Run `eon_driver.py` with a high-uncertainty structure. Assert it exits with the "Halt" code.

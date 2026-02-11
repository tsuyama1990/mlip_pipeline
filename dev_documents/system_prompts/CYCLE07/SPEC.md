# Cycle 07 Specification: Advanced Dynamics (Deposition & aKMC)

## 1. Summary

Cycle 07 extends the capabilities of the **Dynamics Engine** to handle complex, multi-scale simulation scenarios mandated by the User Acceptance Test (Fe/Pt Deposition on MgO). It introduces two advanced features:
1.  **Deposition Simulation**: Support for `fix deposit` in LAMMPS to simulate the growth of thin films and nanoparticles.
2.  **Adaptive Kinetic Monte Carlo (aKMC)**: Integration with the **EON** software package to simulate long-timescale phenomena (ordering, diffusion) that are inaccessible to standard MD.

This cycle addresses the "Time-Scale Problem" by bridging the gap between fast vibrational dynamics (MD) and slow activated events (kMC), using the same ACE potential for both.

## 2. System Architecture

We expand `components/dynamics` to include EON integration and enhanced LAMMPS generation.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```
.
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   └── **config.py**      # Add EONConfig, DepositionConfig
│       └── components/
│           ├── **dynamics/**
│           │   ├── **eon_wrapper.py** # Interface to EON Client
│           │   ├── **eon_driver.py**  # Python script for EON to call (Potentials)
│           │   └── **input_gen.py**   # Update for `fix deposit`
│           └── **factory.py**         # Update for EON support
```

### Key Components
1.  **EONWrapper (`src/mlip_autopipec/components/dynamics/eon_wrapper.py`)**: Manages the EON simulation. It sets up the directory structure, writes `config.ini`, and executes `eonclient`. It monitors the EON process for potential extrapolation errors (OTF support within kMC).
2.  **EON Driver (`src/mlip_autopipec/components/dynamics/eon_driver.py`)**: A standalone script that EON calls to compute energy and forces. This script loads the `.yace` potential (via `pyace` or `lammps`) and returns the values to EON via standard output. Crucially, it also checks $\gamma$ (uncertainty) and can trigger a "Halt" signal to the wrapper.
3.  **Deposition Logic**: Updates to `input_gen.py` to support `fix deposit`, allowing atoms to be inserted at regular intervals with specified velocities and temperatures.

## 3. Design Architecture

### 3.1. Domain Models
*   **DepositionConfig**:
    *   `species`: List of elements to deposit (e.g., ["Fe", "Pt"]).
    *   `rate`: Steps between insertions (e.g., 1000).
    *   `temperature`: Temperature of inserted atoms.
    *   `velocity`: Initial velocity (usually negative z-direction).
    *   `region`: Box dimensions for insertion.
*   **EONConfig**:
    *   `temperature`: KMC temperature (usually lower than MD).
    *   `process_search_method`: "dimer" or "neb".
    *   `searches_per_step`: Number of saddle point searches.

### 3.2. EON Integration Strategy
EON acts as a client-server system, but we run it in "Client Mode" locally.
*   **Communication**: EON executes an external command (our `eon_driver.py`) to get forces.
*   **Driver Logic**:
    1.  Read coordinates from stdin (EON format).
    2.  Compute E, F using ACE potential.
    3.  Compute $\gamma$ (max extrapolation grade).
    4.  If $\gamma > \text{thresh}$, exit with a specific error code (e.g., 100).
    5.  Else, print E, F to stdout.

### 3.3. Deposition (MD) Logic
The `InputGenerator` must construct a loop in LAMMPS:
```lammps
region dep_region block ...
group dep_atoms type ...
fix dep all deposit 100 1 100 12345 region dep_region near 1.0 ...
run 10000
```
This requires careful handling of atom types and groups to distinguish substrate from deposited atoms.

## 4. Implementation Approach

1.  **Dependencies**: `eon` (if available via apt/conda). If not, we mock the `eonclient` execution.
2.  **EON Driver**: Write `eon_driver.py`. This must be a standalone script installable or accessible in the path. It needs `ase` and `pyace` (or `lammps`).
3.  **Wrapper**: Implement `EONWrapper.run()`. It prepares the `config.ini` and calls `subprocess.run(["eonclient"])`.
4.  **Deposition**: Extend `LAMMPSDynamics` to accept `DepositionConfig`.
5.  **Integration**: Update `Orchestrator` to support a workflow step that runs Deposition followed by kMC.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Driver**: Test `eon_driver.py` by piping a sample coordinate file into its stdin and checking stdout. Verify it calculates forces correctly using a mock potential.
*   **Input Generation**: Verify `fix deposit` syntax in `in.lammps`.

### 5.2. Integration Testing (Mock EON)
*   **Goal**: Verify the aKMC loop without compiling EON.
*   **Procedure**:
    1.  Create a dummy `eonclient` script that reads `config.ini` and "finds" a saddle point (returns a new structure).
    2.  Configure the system to run MD (Deposition) -> EON (Relaxation).
    3.  Verify the final structure has changed (ordering increased).
*   **OTF in kMC**:
    1.  Configure the dummy driver to return exit code 100 (Halt).
    2.  Verify the Orchestrator catches this and triggers the "Diagnose" loop.

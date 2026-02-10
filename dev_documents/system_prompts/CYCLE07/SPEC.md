# Cycle 07: Advanced Dynamics (aKMC & Deposition)

## 1. Summary

Cycle 07 expands the simulation capabilities to handle the "Fe/Pt on MgO" use case. This involves two major additions:
1.  **Adaptive Kinetic Monte Carlo (aKMC)**: Integrating the EON software to simulate long-timescale phenomena like ordering and diffusion. Since EON is a C++ code that calls external potentials, we must implement a robust Python driver interface (`pace_driver.py`) that allows EON to use our ACE potential.
2.  **Deposition Module**: Enhancing the `LAMMPSDynamics` to support `fix deposit`. This allows us to simulate the physical process of atoms landing on a substrate, which is critical for the "Interface Learning" phase.

## 2. System Architecture

The following file structure will be modified/created. Files in **bold** are the primary deliverables for this cycle.

```
.
├── config.yaml
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   ├── config.py         # Update DynamicsConfig for EON
│       ├── components/
│       │   ├── dynamics.py       # Update for Deposition
│       │   └── **eon_wrapper.py** # New: EON Integration
│       └── utils/
│           ├── **eon_driver.py** # EON Config Generator
│           └── **pace_driver.py** # The script EON calls
└── tests/
    ├── **test_eon_wrapper.py**
    └── **test_deposition.py**
```

## 3. Design Architecture

### EON Wrapper (`components/eon_wrapper.py`)
This class manages the EON simulation. It implements `BaseDynamics` (or a similar interface).
*   `run_kmc(potential, initial_structure)`:
    1.  Create EON working directory (`.eon/`).
    2.  Write `config.ini` for EON (process search, temperature).
    3.  Write `reactant.con` (initial structure).
    4.  Copy `pace_driver.py` and the potential `.yace` file.
    5.  Run `eonclient`.

### Pacemaker Driver for EON (`utils/pace_driver.py`)
EON expects an executable that reads coordinates from stdin and writes energy/forces to stdout.
*   **Input**: Number of atoms, lattice vectors, coordinates.
*   **Processing**:
    *   Construct `ase.Atoms`.
    *   Calculate Energy & Forces using `pyace` or `lammps` python interface.
    *   **Crucial**: Check Uncertainty ($\gamma$).
*   **Output**:
    *   If $\gamma > \text{threshold}$: Exit with specific error code (Halt).
    *   Else: Print Energy and Forces in EON format.

### Deposition Logic (`components/dynamics.py`)
Update `LAMMPSDynamics` to support a "Deposition Mode".
*   `_generate_deposition_script(rate, temperature, species)`:
    *   Add `region` commands for the substrate and deposition zone.
    *   Add `fix deposit` command.
    *   Ensure the substrate bottom layer is fixed (frozen).

## 4. Implementation Approach

1.  **EON Driver Script**: Implement `utils/pace_driver.py`. This is a standalone script that must be robust. Test it manually by piping in atomic coordinates.
2.  **EON Wrapper**: Implement `components/eon_wrapper.py`.
3.  **Deposition Update**: Add `deposition` parameters to `DynamicsConfig` and implement the logic in `LAMMPSDynamics`.
4.  **Integration**: Update `Orchestrator` to support a "kMC" stage in the workflow.

## 5. Test Strategy

### Unit Testing
*   **Deposition Script**: Generate a LAMMPS script with `deposition_rate=100`. Verify `fix deposit` syntax.
*   **EON Config**: Verify `config.ini` generation.

### Integration Testing (Driver)
*   **pace_driver.py**:
    *   Create a simple structure file.
    *   Run `cat structure | python utils/pace_driver.py`.
    *   Assert output format matches EON requirements (Energy line, then Force lines).

### Integration Testing (EON - Optional)
*   **Mock Run**: If `eonclient` is available, run a 1-step search.
*   **Halt Test**: Configure the driver to return a high uncertainty. Verify EON stops or the wrapper detects the exit code.

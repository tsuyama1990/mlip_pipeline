# Cycle 05 Specification: Dynamics Engine (LAMMPS & Uncertainty)

## 1. Summary

In this cycle, we replace the `MockDynamics` with a functional `LAMMPSDynamics` engine. This module is responsible for running Molecular Dynamics (MD) simulations using the ACE potential. Crucially, it implements two key safety features:
1.  **Hybrid Potential**: Combines ACE with a physics-based baseline (ZBL/LJ) using `pair_style hybrid/overlay`. This prevents non-physical behavior (e.g., nuclear fusion) in extrapolation regions.
2.  **Uncertainty Watchdog**: Uses `fix halt` to terminate the simulation if the model's uncertainty metric ($\gamma$, extrapolation grade) exceeds a threshold. This signal triggers the Active Learning loop to acquire new data.

## 2. System Architecture

The following file structure will be created. **Bold** files are the targets for this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── **lammps.py**           # LAMMPS Interface (via ASE or lammps-python)
│   │   ├── **base.py**             # (Update ABC with halt logic)
│   │   └── **potentials.py**       # Helper for generating pair_style strings
tests/
└── **test_dynamics.py**            # Tests for LAMMPS Execution & Halt
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/config.py`)
*   **`DynamicsConfig`**:
    *   `command`: str (e.g., `lmp_serial`)
    *   `timestep`: float (fs)
    *   `n_steps`: int (MD duration)
    *   `temperature`: float (K)
    *   `uncertainty_threshold`: float (Max allowed $\gamma$, e.g., 5.0)
    *   `baseline_potential`: Enum (ZBL, LJ)

### 3.2. Dynamics Components (`components/dynamics/lammps.py`)
*   **`LAMMPSDynamics`**:
    *   `run(structure, potential) -> DynamicsResult`:
        1.  **Setup**: Initialize LAMMPS instance (using `ase.calculators.lammpsrun` or `lammps` python module).
        2.  **Configure Potential**: Generate `pair_style hybrid/overlay pace zbl` commands using `potentials.py`.
        3.  **Configure Watchdog**:
            *   `compute pace_gamma all pace ... gamma_mode=1`
            *   `fix halt 10 v_max_gamma > {threshold} error hard`
        4.  **Run MD**: Execute `run {n_steps}`.
        5.  **Handle Outcome**:
            *   **Success**: Return `DynamicsResult(halted=False, final_structure=atoms)`.
            *   **Halt**: Catch "Halt triggered" error. Parse log to find the timestep and structure where $\gamma$ exceeded threshold. Return `DynamicsResult(halted=True, halt_structure=bad_atoms)`.

### 3.3. Potential Helpers (`components/dynamics/potentials.py`)
*   **`HybridPotentialGenerator`**:
    *   `generate_pair_style(elements, potential_path, baseline="zbl") -> str`:
        *   Constructs the complex LAMMPS string.
        *   Example: `pair_style hybrid/overlay pace zbl 1.0 2.0`
        *   Example: `pair_coeff * * pace {path} {elements}`
        *   Example: `pair_coeff * * zbl {zbl_params}`

## 4. Implementation Approach

1.  **Dynamics Config**: Define `DynamicsConfig` and update `config.py`.
2.  **Potential Logic**: Implement `HybridPotentialGenerator`. Verify correct syntax for `hybrid/overlay`.
3.  **LAMMPS Interface**: Implement `LAMMPSDynamics`.
    *   Use `ase.io.write(..., format='lammps-data')` to generate data files.
    *   Use `subprocess` or `lammps` library to run.
    *   **Crucial**: Implement log parsing to extract the "bad" structure upon halt.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_potentials.py`**:
    *   Verify that `generate_pair_style` produces valid LAMMPS input strings for a given list of elements.

### 5.2. Integration Testing (Mocked Binary)
*   **`test_dynamics.py`**:
    *   **Mock `lammps.run`**: Simulate a halt condition (e.g., return specific error string).
    *   **Verify Halt Handling**: Assert that `DynamicsResult.halted` is `True` and `halt_structure` is not None.
    *   **Verify Success**: Simulate a clean run. Assert `halted` is `False`.
    *   **Verify Hybrid**: Check generated input file (`in.lammps`) contains `pair_style hybrid/overlay`.

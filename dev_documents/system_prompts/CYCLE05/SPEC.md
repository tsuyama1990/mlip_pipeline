# Cycle 05 Specification: Dynamics Engine (LAMMPS MD)

## 1. Summary
**Goal**: Implement the `DynamicsEngine` for Molecular Dynamics (MD) using LAMMPS. This component is the primary source of exploration data. It must safely run simulations using a "Hybrid Potential" (ACE + ZBL) and implement a "Watchdog" to halt simulations when uncertainty ($\gamma$) spikes.

**Key Features**:
*   **LAMMPS Interface**: Generate `in.lammps` files and run via `subprocess` or Python API (`lammps` module).
*   **Hybrid Potential**: Automatically configure `pair_style hybrid/overlay pace zbl` to prevent nuclear fusion.
*   **Uncertainty Watchdog**: Implement `fix halt` logic to stop simulations when `max(gamma) > threshold`.
*   **Halt & Diagnose**: Return the exact snapshot and atom IDs that caused the halt for targeted learning.

## 2. System Architecture

Files to be implemented/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── **dynamics.py**         # MD Config & Halt Info
│   └── ...
├── dynamics/
│   ├── **__init__.py**
│   ├── **base.py**             # Abstract Base Class
│   ├── **md_driver.py**        # LAMMPS Driver
│   └── **halt_handler.py**     # Diagnose Halt Snapshots
└── tests/
    └── **test_dynamics/**
        └── **test_md_driver.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`src/mlip_autopipec/domain_models/dynamics.py`)

*   **`MDConfig`**:
    *   `ensemble`: `NVT`, `NPT`, `NVE`.
    *   `temperature`: (start, end).
    *   `pressure`: (start, end).
    *   `steps`: Total steps (int).
    *   `uncertainty_threshold`: Max allowed $\gamma$ (float, e.g., 5.0).
*   **`MDResult`**:
    *   `final_structure`: Structure.
    *   `trajectory`: List[Structure] (optional).
    *   `halted`: bool.
    *   `halt_reason`: str ("Max Steps" or "High Uncertainty").
    *   `max_gamma`: float.

### 3.2. Dynamics Component (`src/mlip_autopipec/dynamics/`)

#### `base.py`
*   **`BaseDynamics`** (ABC):
    *   `run(structure: Structure, potential: Path) -> MDResult`

#### `md_driver.py`
*   **`LAMMPSDriver`**:
    *   Generates `in.lammps`.
    *   **Crucial Lines**:
        ```
        pair_style hybrid/overlay pace zbl 1.0 2.0
        pair_coeff * * pace potential.yace elements...
        pair_coeff * * zbl atomic_numbers...
        compute pace_gamma all pace ... gamma_mode=1
        variable max_gamma equal max(c_pace_gamma)
        fix watchdog all halt 10 v_max_gamma > ${THRESHOLD} error hard
        ```
    *   Parses `log.lammps` to detect if "error hard" was triggered (Halt) or normal completion.

#### `halt_handler.py`
*   **`HaltDiagnoser`**:
    *   If halted, reads the `dump.lammps` file.
    *   Extracts the *last* frame (the high-uncertainty configuration).
    *   Identifies atoms with high $\gamma$.
    *   Returns a list of `Structure` objects centered on these atoms (Candidates).

## 4. Implementation Approach

1.  **Define Interfaces**: Create `base.py`.
2.  **Implement LAMMPS Driver**: Write `in.lammps` template generation using Jinja2 or f-strings.
3.  **Implement Watchdog Logic**: Ensure `fix halt` works as expected.
4.  **Implement Halt Handler**: Use `ase.io.read` to parse trajectories.
5.  **Mock LAMMPS**: If `lmp` binary is missing, simulate a run that halts after N steps with a fake high-gamma snapshot.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_md_driver.py`**:
    *   Generate input file for a dummy structure.
    *   Verify `pair_style hybrid/overlay` is present.
    *   Verify `fix halt` command uses the correct variable and threshold.

### 5.2. Integration Testing
*   **Mock Run**:
    *   Mock `subprocess.run` to simulate LAMMPS exit code.
    *   Provide a dummy `log.lammps` indicating a halt.
    *   Assert `MDResult.halted` is True.

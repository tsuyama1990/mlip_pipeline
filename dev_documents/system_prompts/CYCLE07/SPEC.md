# Cycle 07 Specification: Advanced Dynamics (EON & Deposition)

## 1. Summary

Cycle 07 addresses the "Time-Scale Problem" and complex material synthesis workflows. We integrate **EON** (Adaptive Kinetic Monte Carlo) to allow the system to explore rare events (diffusion, reactions) that occur on timescales far beyond the reach of standard MD.

Additionally, we implement a **Deposition Module** within the LAMMPS driver. This allows simulating "Physical Vapor Deposition" (PVD) or "Epitaxial Growth" by periodically inserting atoms into the simulation box. This is critical for the "Fe/Pt on MgO" user scenario.

## 2. System Architecture

### File Structure

Files in **bold** are new or modified in this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── **eon_driver.py**    # EON client wrapper
│   │   ├── **deposition.py**    # Logic for generating 'fix deposit' inputs
│   │   └── **scripts/**         # Templates
│   │       └── **pace_driver.py** # Script called by EON to evaluate potential
│   └── mock.py
├── domain_models/
│   └── config.py                # Updated with EONConfig and DepositionConfig
└── core/
    └── orchestrator.py
```

### Component Interaction (EON)
1.  **Orchestrator** decides to run kMC (e.g., after an MD relaxation).
2.  **EON Driver** sets up the `reactant.con` (initial structure) and `config.ini`.
3.  **EON Driver** writes `potentials/pace_driver.py`, which wraps the ACE potential for EON.
4.  **EON Driver** launches `eonclient`.
5.  **pace_driver.py** (running inside EON's process loop):
    -   Reads coordinates from stdin.
    -   Calculates Energy/Force using `pyacemaker`.
    -   **Checks Uncertainty ($\gamma$)**: If high, it returns a special error code to EON.
6.  **EON Driver** detects the error, extracts the high-uncertainty structure, and returns "Halt" to Orchestrator.

### Component Interaction (Deposition)
1.  **Deposition Config**: Specifies `rate`, `elements`, `temperature`.
2.  **Input Generator**: Adds `fix deposit` commands to the LAMMPS script.
    -   `fix 1 all deposit 100 0 100 12345 region my_region near 1.0 ...`
3.  **Execution**: Runs standard MD, but atoms are added over time.

## 3. Design Architecture

### 3.1. EON Configuration (`domain_models/config.py`)

```python
class EONConfig(BaseComponentConfig):
    enabled: bool = False
    temperature: float = 300.0
    process_search_method: Literal["dimer", "neb"] = "dimer"
    max_steps: int = 100
```

### 3.2. Deposition Configuration

```python
class DepositionConfig(BaseModel):
    species: List[str]
    rate: float  # atoms per ps
    total_atoms: int
    insert_region: List[float]  # [z_min, z_max]
```

### 3.3. The EON Driver Script (`components/dynamics/scripts/pace_driver.py`)

This script acts as the bridge. It must be standalone executable.

```python
# Template
import sys
from ase.io import read
from mlip_autopipec.calculators import PaceCalculator

def main():
    atoms = read(sys.stdin, format='con')
    calc = PaceCalculator("potential.yace")

    gamma = calc.get_gamma(atoms)
    if gamma > 5.0:
        sys.exit(100)  # Signal Halt

    e = calc.get_potential_energy(atoms)
    f = calc.get_forces(atoms)

    print(f"Energy: {e}")
    # ... print forces ...
```

## 4. Implementation Approach

1.  **Implement Deposition Logic**: Extend `input_gen.py` in Cycle 05 to accept `DepositionConfig` and write `fix deposit` lines.
2.  **Implement EON Wrapper**:
    -   Create `eon_driver.py`.
    -   Implement logic to write `config.ini` and `reactant.con`.
    -   Use `subprocess` to run `eonclient`.
3.  **Implement Potential Driver**: Create the template for `pace_driver.py`.
4.  **Mocking**:
    -   **Mock EON**: Simulate a kMC run that either finishes (returns lower energy structure) or halts (high uncertainty).
    -   **Mock Deposition**: Simulate an MD run where the number of atoms increases.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Deposition Input**: Verify `fix deposit` syntax in generated LAMMPS script.
-   **EON Config**: Verify `config.ini` generation.

### 5.2. Integration Testing
-   **EON Loop (Mock)**: Run the Orchestrator with `dynamics.mode: eon`. Verify it handles the EON return codes.
-   **Real EON (Optional)**: If `eonclient` is installed, run a trivial search (e.g., adatom diffusion).
-   **Deposition MD**: Run a mock deposition and verify the final structure has `N_initial + N_deposited` atoms.

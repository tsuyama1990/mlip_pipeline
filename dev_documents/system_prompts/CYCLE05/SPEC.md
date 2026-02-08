# Cycle 05: Dynamics Engine (OTF Loop)

## 1. Summary

Cycle 05 delivers the **Dynamics Engine**, the component responsible for exploring the phase space using the trained potential. This engine runs Molecular Dynamics (MD) via LAMMPS and Adaptive Kinetic Monte Carlo (aKMC) via EON.

The critical feature here is **On-the-Fly (OTF) Active Learning**. The engine does not blindly run simulations. Instead, it continuously monitors the **extrapolation grade ($\gamma$)** of every atomic environment. If the simulation encounters a configuration where the potential is uncertain ($\gamma > \gamma_{thresh}$), it immediately halts and returns the "dangerous" structure to the Orchestrator for labeling. This prevents the "garbage in, garbage out" problem and ensures the potential is always operating within its valid domain.

Furthermore, we enforce **Physics-Informed Safety** by using a **Hybrid Potential**. The engine automatically configures LAMMPS to overlay a stiff core-repulsion potential (ZBL or LJ) on top of the machine-learned ACE potential. This guarantees that atoms never overlap, even if the ML model predicts unphysical attraction at short distances.

## 2. System Architecture

Files in **bold** are new or modified.

```ascii
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── **lammps.py**         # LAMMPS Integration
│   │   ├── **eon.py**            # EON (kMC) Integration
│   │   ├── **otf.py**            # Uncertainty Monitoring Logic
│   │   └── **hybrid.py**         # Hybrid Potential Configuration
│   └── ...
```

## 3. Design Architecture

### 3.1. LAMMPS Integration (`lammps.py`)
-   **Class `LAMMPSDriver`**:
    -   `run_md(structure: Structure, potential: Potential, settings: MDSettings) -> MDResult`
    -   **Input Generation**: Writes `in.lammps`, `data.lammps`.
    -   **Execution**: Calls `lmp` binary via `subprocess`.
    -   **OTF Monitoring**: Includes `compute pace` and `fix halt` commands.
    -   **Result Parsing**: Reads `log.lammps` to determine if halted or finished. Returns the final structure (or halted snapshot).

### 3.2. Hybrid Potential Logic (`hybrid.py`)
-   **Function `generate_pair_style(potential: Potential, baseline: BaselineConfig) -> str`**:
    -   Constructs the `pair_style hybrid/overlay pace zbl ...` string.
    -   Calculates the inner/outer cutoffs for the switching function to ensure smooth transition.

### 3.3. EON Integration (`eon.py`)
-   **Class `EONDriver`**:
    -   `run_kmc(structure: Structure, potential: Potential, settings: KMCSettings) -> KMCResult`
    -   **Setup**: Creates `config.ini`, `pos.con`.
    -   **Driver Script**: Generates a python script (`pace_driver.py`) that EON calls to get energy/forces from the ACE potential.
    -   **OTF**: The driver script checks $\gamma$ and returns a special exit code to abort EON if uncertainty is high.

## 4. Implementation Approach

1.  **Refactor**: Ensure `Potential` object knows its element mapping.
2.  **Hybrid Logic**: Implement `generate_pair_style` in `hybrid.py`.
3.  **LAMMPS**: Implement `LAMMPSDriver`.
    -   Use `jinja2` templates for `in.lammps`.
    -   Add `fix halt` logic: `fix halt all halt 10 v_max_gamma > ${thresh} error hard`.
4.  **EON**: Implement `EONDriver` (if binary available, else Mock).
    -   Focus on the Python driver interface (`pace_driver.py`) which allows EON to use *any* Python calculator.
5.  **Integration**: Update `Orchestrator` to handle `MDResult` (detect halt -> extract structure -> add to dataset).

## 5. Test Strategy

### 5.1. Unit Tests
-   **Input Generation**: Verify `in.lammps` contains `pair_style hybrid/overlay`.
-   **OTF Logic**: Verify the threshold variable is correctly set in the input script.

### 5.2. Integration Tests (Mock LAMMPS)
-   **Mock Binary**: Create a dummy `lmp` script that:
    -   Reads input.
    -   Writes a `log.lammps`.
    -   Writes a `dump.lammps` with random positions.
    -   (Optionally) Simulates a "Halt" by writing "ERROR: Halt" to stderr.
-   **Run**: `Dynamics.run_md()`. Verify it parses the halt correctly.

### 5.3. Real MD Test (Requires LAMMPS + USER-PACE)
-   **Test Case**: Run NVE dynamics on Bulk Cu for 100 steps.
-   **Validation**: Energy conservation.
-   **Halt Test**: Set threshold $\gamma=0.0$ (impossible). Verify simulation halts immediately at step 0.

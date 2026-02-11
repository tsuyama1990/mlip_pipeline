# Cycle 03 Specification: Oracle (DFT & Self-Healing)

## 1. Summary
This cycle implements the "Judge" component: the `Oracle`. Its primary responsibility is to calculate the ground-truth potential energy surface (Energy, Forces, Virial Stress) for the candidate structures generated in the previous cycle. This module is built around reliability. It encapsulates the complexity of Density Functional Theory (DFT) codes (specifically Quantum Espresso) and implements a "Self-Healing" mechanism. If a calculation fails due to common issues (SCF non-convergence), the Oracle detects the error, modifies the input parameters (e.g., reducing mixing beta), and retries automatically. It also implements "Periodic Embedding," a technique to efficiently calculate forces for local clusters carved out of large MD simulations.

## 2. System Architecture

### 2.1. File Structure
files to be created/modified in this cycle are bolded.

```text
src/mlip_autopipec/
├── components/
│   ├── oracle/
│   │   ├── __init__.py
│   │   ├── base.py                 # [CREATE] Abstract Base Class
│   │   ├── qe_driver.py            # [CREATE] Quantum Espresso Wrapper
│   │   ├── input_gen.py            # [CREATE] Input File Factory
│   │   ├── self_healer.py          # [CREATE] Error Recovery Logic
│   │   └── embedding.py            # [CREATE] Periodic Embedding Logic
├── domain_models/
│   ├── config.py                   # [MODIFY] Add DFTConfig
│   └── enums.py                    # [MODIFY] Add DFTStatus, OracleType
└── core/
    └── orchestrator.py             # [MODIFY] Integrate Oracle into label()
```

### 2.2. Component Interaction
1.  **`Orchestrator`** calls `oracle.compute(structures, context=state)`.
2.  **`Oracle`** loops through structures:
    *   **`PeriodicEmbedding`**: If structure is a cluster (non-periodic), wraps it in a box.
    *   **`InputGenerator`**: Creates `pw.x` input file (selecting pseudos, k-points).
    *   **`QEDriver`**: Launches `pw.x` process.
    *   **`SelfHealer`** (Monitor):
        *   Checks output log for "convergence NOT achieved".
        *   If failed, modifies params and triggers retry.
        *   If success, parses output (ASE compatible).
3.  Returns list of `LabeledStructure` (Atoms with `calc` attached).

## 3. Design Architecture

### 3.1. Domain Models

#### `enums.py`
*   `OracleType`: `MOCK`, `QE`, `VASP`.
*   `DFTStatus`: `PENDING`, `RUNNING`, `CONVERGED`, `FAILED_RETRYABLE`, `FAILED_FATAL`.

#### `config.py`
*   `DFTConfig`:
    *   `command`: str (e.g., `mpirun -np 4 pw.x`)
    *   `pseudopotentials`: Dict[str, str] (Path to UPF)
    *   `kspacing`: float (e.g., 0.04)
    *   `max_retries`: int

### 3.2. Core Logic

#### `self_healer.py`
*   **Responsibility**: Finite State Machine for error recovery.
*   **States**:
    *   `Attempt 1`: Standard settings.
    *   `Attempt 2`: `mixing_beta` * 0.5.
    *   `Attempt 3`: `electron_maxstep` * 2.
    *   `Attempt 4`: `smearing` = 'mv', `degauss` * 2.
*   **Logic**: Parses standard error streams or output files to identify error codes.

#### `embedding.py`
*   **Responsibility**: Prepare clusters for periodic DFT.
*   **Logic**:
    *   Input: `Atoms` (Cluster with void).
    *   Action: Define bounding box + `vacuum_padding`. Create new unit cell. Center atoms.

## 4. Implementation Approach

### Step 1: Interface Definition
*   Define `BaseOracle` in `components/oracle/base.py`.
*   Define `compute(self, atoms: List[Atoms]) -> List[Atoms]`.

### Step 2: Input Generation
*   Implement `InputGenerator`.
*   Use `ase.calculators.espresso` or string templating.
*   **Critical**: Ensure `tprnfor=.true.` and `tstress=.true.` are always set.

### Step 3: Self-Healing Logic
*   Implement `SelfHealer`.
*   Create a "Mock QE" for testing that fails N times before succeeding.

### Step 4: Periodic Embedding
*   Implement `embedding.py`.
*   Ensure atom mapping is preserved (which atom in supercell corresponds to input atom?).

### Step 5: Orchestrator Integration
*   Update `Orchestrator.label()` to call the Oracle.
*   Save the labeled structures to `work_dir/iter_XX/dft_calc/`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_input_gen.py`**:
    *   Verify generated text contains `K_POINTS automatic`.
    *   Verify pseudopotentials are correctly mapped.
*   **`test_self_healer.py`**:
    *   Feed a log file with "convergence NOT achieved".
    *   Assert that `suggest_fix()` returns a config with reduced beta.

### 5.2. Integration Testing
*   **`test_oracle_mock.py`**:
    *   Configure `Oracle` with `MOCK` type.
    *   Pass a list of atoms.
    *   Assert that returned atoms have random forces attached.
*   **`test_oracle_qe_dryrun.py` (Optional)**:
    *   If `pw.x` is available, run a tiny H2 molecule calculation.
    *   Verify `atoms.get_forces()` returns valid array.

# Cycle 06 Specification: Validation & Full Orchestration

## 1. Summary
Cycle 06 is the final integration phase. We implement the `Validator` component to perform rigorous physical tests (Phonons, Elastic Constants, EOS) on the trained potential. We also finalize the `Orchestrator` to seamlessly connect all components (Generator -> Oracle -> Trainer -> Dynamics -> Validator) into a continuous active learning loop. Finally, we polish the CLI for production use.

## 2. System Architecture

### 2.1 File Structure
**Bold** files are to be created or modified in this cycle.

```ascii
.
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   └── **orchestrator.py**      # Final Full Loop Logic
│       ├── components/
│       │   ├── validator/
│       │   │   ├── **base.py**          # Base Validator
│       │   │   ├── **phonon.py**        # Phonon Stability Test
│       │   │   ├── **elastic.py**       # Elastic Constants Test
│       │   │   └── **eos.py**           # Equation of State Test
│       ├── domain_models/
│       │   └── **validator_config.py**  # Config for validation tests
│       └── utils/
│           └── **report_generator.py**  # HTML/Markdown Report Generator
```

## 3. Design Architecture

### 3.1 Validator Component (`src/mlip_autopipec/components/validator/`)

*   **`Validator` Class**:
    *   Runs a suite of tests on the `Potential`.
    *   Returns a `ValidationReport` object (pass/fail status, metrics, plots).

*   **Tests**:
    1.  **Phonon Stability**:
        *   Calculates phonon dispersion (using `phonopy` or `ase.phonons`).
        *   Checks for imaginary frequencies ($\omega^2 < 0$) which indicate dynamic instability.
    2.  **Elastic Constants**:
        *   Calculates $C_{ij}$ matrix by straining the cell.
        *   Checks Born stability criteria (e.g., $C_{11} - C_{12} > 0$).
    3.  **EOS**:
        *   Fits Energy-Volume curve to Birch-Murnaghan equation.
        *   Checks bulk modulus $B_0$.

### 3.2 Orchestrator Logic Finalization
The `Orchestrator` must now handle the full complex loop:
1.  **Exploration**: Run MD/kMC.
2.  **Detection**: If MD halts (OTF), extract structures.
3.  **Selection**: If data is too large, select Active Set.
4.  **Calculation**: Send to Oracle (DFT).
5.  **Refinement**: Train new potential.
6.  **Validation**: Run Validator.
    *   If PASS: Deploy to `production/`.
    *   If FAIL: Feedback loop (e.g., add the failed structure to training set with higher weight, or request more data in that region).

## 4. Implementation Approach

1.  **Validator**: Implement `elastic.py` and `eos.py` using ASE. (Phonopy integration might be optional if dependencies are heavy, or mocked).
2.  **Orchestrator**: Connect the "Halted MD" output from Cycle 05 to the "Oracle" input of Cycle 03.
3.  **Reporting**: Create a simple HTML report generator that compiles plots and metrics.
4.  **CLI**: Ensure `mlip-pipeline run config.yaml` works end-to-end.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Elastic**: Test on a known potential (e.g., LJ argon) where $C_{ij}$ values are known.
*   **EOS**: Test fitting logic with perfect B-M data.

### 5.2 Integration Testing
*   **Full Cycle (Mocked)**: Run the entire loop with Mock components but *Real* data flow logic.
    *   Generator -> Dataset -> Trainer -> Potential -> Dynamics (Halt) -> Oracle -> Dataset -> Trainer -> Validator.
*   **Validation Gate**: Verify that if Validator fails, the Orchestrator logs a warning and does not mark the cycle as "Success".

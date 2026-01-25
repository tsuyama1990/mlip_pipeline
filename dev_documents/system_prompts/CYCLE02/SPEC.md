# Cycle 02: Oracle (DFT Engine)

## 1. Summary
This cycle implements the **Oracle** module, responsible for generating ground-truth data using First-Principles calculations (DFT). We focus on integrating **Quantum Espresso (QE)** as the primary backend. The module must be robust, capable of handling SCF convergence failures automatically ("Self-Healing"), and efficient, managing batch calculations for the Active Learning loop.

## 2. System Architecture

We add the `phases/dft` module and its components.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── orchestration/
│   └── phases/
│       ├── **base.py**              # Abstract Base Class for phases
│       └── **dft/**
│           ├── **__init__.py**
│           ├── **runner.py**        # QE Execution Wrapper
│           ├── **input_gen.py**     # Input file generation (ASE adapter)
│           ├── **recovery.py**      # Error handling & self-correction
│           └── **manager.py**       # DFTPhase implementation
└── tests/
    └── **test_dft.py**
```

## 3. Design Architecture

### The Oracle Interface
The `DFTPhase` class implements the standard phase interface, accepting a list of `CandidateStructure` objects and returning `LabeledStructure` objects (Energy, Forces, Stress).

### Quantum Espresso Runner (`runner.py`)
*   **Command Execution**: Wraps `subprocess.run` to call `pw.x`.
*   **Environment**: Handles MPI settings (`mpirun`) and environment variables (pseudo-potentials).

### Self-Healing Mechanism (`recovery.py`)
A state machine that attempts to fix convergence errors.
*   **Strategy 1 (Default)**: Standard mixing and smearing.
*   **Strategy 2 (Soft)**: Reduce `mixing_beta` (0.7 -> 0.3).
*   **Strategy 3 (Hot)**: Increase electronic temperature (`smearing`).
*   **Strategy 4 (Robust)**: Change diagonalization (David -> CG).

## 4. Implementation Approach

1.  **Phase Interface**: Define the `Phase` abstract base class in `phases/base.py`.
2.  **Input Generation**: Use `ase.calculators.espresso` to generate standard QE input files. Enforce `tprnfor=.true.` and `tstress=.true.` to ensure forces and stress are calculated.
3.  **Runner Implementation**: Create `QERunner` to execute the command. Implement output parsing to detect "Convergence NOT achieved" or crash errors.
4.  **Recovery Logic**: Implement `RecoveryHandler`. It should take a failed calculation, modify the parameters, and retry.
5.  **Manager Integration**: `DFTPhase.run()` should accept a batch of structures, loop through them (possibly in parallel), and return results.

## 5. Test Strategy

### Unit Testing
*   **`test_input_gen.py`**: Verify that generated inputs contain required flags (`tprnfor`, `tstress`).
*   **`test_recovery.py`**: Feed fake error logs to the handler and verify it suggests the correct parameter adjustments.

### Integration Testing
*   **Mock QE**: Create a "Mock Runner" that simulates `pw.x` behavior.
    *   If input has `mixing_beta > 0.5`, return "Convergence NOT achieved".
    *   If input is correct, return a fixed energy/force output.
*   **End-to-End**: Run `DFTPhase` with the mock runner to verify the retry loop works (Fail -> Adjust -> Success).

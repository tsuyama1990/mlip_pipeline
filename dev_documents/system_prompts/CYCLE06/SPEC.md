# Cycle 06 Specification: Validation, Integration & Orchestration

## 1. Summary
This cycle completes the system by implementing the `Validator` component (Phonons, Elastic Constants, EOS) and finalizing the `Orchestrator` logic to fully integrate all previous components. This is the "Polish & QA" phase where the system transitions from a collection of parts to a cohesive application. The goal is to successfully execute the Fe/Pt on MgO scenario from start to finish.

## 2. System Architecture

### 2.1. File Structure
The following files must be created or modified. **Bold** files are the focus of this cycle.

src/mlip_autopipec/
├── domain_models/
│   ├── **validation.py**           # ValidationResult, Metric
├── interfaces/
│   ├── **validator.py**            # Enhanced BaseValidator
├── infrastructure/
│   ├── **validator/**
│   │   ├── **__init__.py**
│   │   ├── **phonon.py**           # Phonon Calculator
│   │   ├── **elastic.py**          # Elastic Constants
│   │   └── **eos.py**              # Equation of State
├── orchestrator/
│   ├── **full_orchestrator.py**    # Full Integration Logic (replacing Simple)
│   └── **error_handling.py**       # Robust Exception Handling
└── **main.py**                     # Enhanced CLI (subcommands: init, run, validate)

### 2.2. Class Diagram
*   `PhononValidator` checks dynamical stability.
*   `ElasticValidator` checks mechanical stability (Born Criteria).
*   `FullOrchestrator` manages the complex state transitions (Explore -> Halt -> Label -> Train -> Validate).

## 3. Design Architecture

### 3.1. Validator Logic (`infrastructure/validator/phonon.py`)
*   **Input**: `Potential`, `Structure` (Relaxed).
*   **Process**:
    1.  Create supercell (e.g., 2x2x2).
    2.  Calculate Force Constants (Phonopy integration or Finite Displacement).
    3.  Compute Band Structure.
    4.  Check for Imaginary Frequencies (Instability).
*   **Output**: `ValidationResult` (Passed/Failed, Max imaginary freq).

### 3.2. Validator Logic (`infrastructure/validator/elastic.py`)
*   **Input**: `Potential`, `Structure` (Relaxed).
*   **Process**:
    1.  Apply strains (-1%, +1%) to unit cell.
    2.  Compute Stress Tensor.
    3.  Fit to obtain Cij Matrix.
    4.  Check Born Stability Criteria (Positive Definite).
*   **Output**: `ValidationResult` (Passed/Failed, Bulk Modulus).

### 3.3. Full Orchestrator (`orchestrator/full_orchestrator.py`)
*   **Goal**: robustly manage the loop, handling:
    *   DFT Convergence Failures (Self-Healing).
    *   MD Halts (OTF Learning).
    *   Validation Failures (Re-training or Warning).
    *   Checkpointing (Save state to disk to resume later).

## 4. Implementation Approach

1.  **Dependencies**: `phonopy` (optional but recommended), `scipy`.
2.  **Implement Validators**: Create the validation classes.
3.  **Implement Orchestrator**: Upgrade the loop logic.
4.  **Implement CLI**: Add `validate` command to run checks on existing potentials.
5.  **Update Config**: Add `ValidatorConfig` (phonon supercell size, strain amount).

## 5. Test Strategy

### 5.1. Unit Testing
*   **Elastic**: Verify C11, C12 calculation on a known potential (e.g., LJ) matches analytical values.
*   **Orchestrator**: Verify state machine transitions correctly (e.g., if Validation fails, does it retry or stop?).

### 5.2. Integration Testing
*   **Full Fe/Pt on MgO Run**:
    *   Run the complete pipeline on a small scale (Mock Mode or Tiny System).
    *   Verify all steps execute in order.
    *   Verify final potential passes basic validation checks.

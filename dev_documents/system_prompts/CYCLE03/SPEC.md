# Cycle 03 Specification: The Oracle (DFT Automation)

## 1. Summary
This cycle implements the `Oracle` component, the interface to high-fidelity physics engines (specifically Quantum Espresso via ASE). The goal is to automate the calculation of energy, forces, and stresses for any given atomic structure. Key features include automatic input generation (k-spacing, pseudopotentials), self-healing mechanisms for SCF convergence failures, and "Periodic Embedding" to extract clusters from larger simulations for efficient labeling.

## 2. System Architecture

### 2.1. File Structure
The following files must be created or modified. **Bold** files are the focus of this cycle.

src/mlip_autopipec/
├── domain_models/
│   ├── **structure.py**            # Periodic Embedding Support
│   ├── **config.py**               # OracleConfig (QEScConfig)
├── interfaces/
│   ├── **oracle.py**               # Enhanced BaseOracle
├── infrastructure/
│   ├── **oracle/**
│   │   ├── **__init__.py**
│   │   ├── **qe_oracle.py**        # Quantum Espresso Implementation
│   │   ├── **dft_manager.py**      # High-level DFT orchestration
│   │   ├── **self_healing.py**     # Error handling & Recovery Logic
│   │   └── **embedding.py**        # Periodic Embedding Logic
└── orchestrator/
    └── **simple_orchestrator.py**  # Update logic to call Oracle

### 2.2. Class Diagram
*   `QEOracle` implements `BaseOracle`.
*   `DFTManager` handles job submission, monitoring, and result parsing.
*   `SelfHealing` analyzes errors and modifies parameters (mixing beta, smearing).

## 3. Design Architecture

### 3.1. Oracle Logic (`infrastructure/oracle/qe_oracle.py`)
*   **Input**: `Iterable[Structure]`
*   **Process**:
    1.  Validate inputs (Elements supported by Pseudopotential library?).
    2.  For each structure:
        *   Generate `pw.x` input file (control, system, electrons).
        *   Set `kpts` based on `kspacing` (e.g., 0.04 1/Å).
        *   Set `pseudopotentials` using SSSP (Efficiency/Precision).
        *   Run `pw.x` via ASE `Espresso` calculator or `subprocess`.
        *   Parse output XML/log for Energy, Forces, Stress.
*   **Output**: `Iterator[Structure]` with updated properties.

### 3.2. Self-Healing (`infrastructure/oracle/self_healing.py`)
*   **Logic**:
    *   If `SCF NOT CONVERGED`:
        *   Attempt 1: Reduce `mixing_beta` (0.7 -> 0.3).
        *   Attempt 2: Increase `electron_maxstep` (100 -> 200).
        *   Attempt 3: Increase `smearing` (0.01 -> 0.02 Ry).
    *   If `Crash`: Log error and skip structure (return None or flagged structure).

### 3.3. Periodic Embedding (`infrastructure/oracle/embedding.py`)
*   **Goal**: Isolate a cluster from a large MD snapshot for DFT calculation while maintaining periodic boundary conditions.
*   **Logic**:
    *   Identify "Active Region" (e.g., high uncertainty atoms).
    *   Identify "Buffer Region" (surrounding atoms within cutoff).
    *   Create a new minimal Orthorhombic cell containing Active + Buffer.
    *   Mask forces on Buffer atoms (set weight = 0 during training).

## 4. Implementation Approach

1.  **Dependencies**: Ensure `ase` is installed. Check for `pw.x` binary (or mock wrapper).
2.  **Implement DFTManager**: Create the core class to manage ASE calculators.
3.  **Implement QEOracle**: Wrap `DFTManager` for Quantum Espresso specifics.
4.  **Implement Self-Healing**: Add try-except blocks around `calc.get_potential_energy()`.
5.  **Implement Embedding**: Add geometric manipulation functions using `ase`.
6.  **Update Config**: Add `OracleConfig` for QE paths and parameters.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Input Generation**: Verify `QEOracle` generates correct `pw.in` text (k-points, pseudos).
*   **Parsing**: Verify `QEOracle` correctly parses a sample `pw.out` file (mocked stdout).
*   **Healing Logic**: Simulate a `SCFError` exception and verify `mixing_beta` is updated in the next attempt.

### 5.2. Integration Testing
*   **Real/Mock Calculation**:
    *   Run `QEOracle` on a simple Silicon primitive cell.
    *   If `pw.x` is available: Check energy is approx -308 eV (pseudo dependent).
    *   If `pw.x` is missing: Use a "Fake Binary" script that outputs pre-computed text.
    *   Assert Forces are close to zero for relaxed structure.

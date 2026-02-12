# Cycle 03 Specification: Oracle (DFT Automation)

## 1. Summary
**Goal**: Implement the `Oracle` component, responsible for running First-Principles calculations (DFT) to generate ground-truth data (Energy, Forces, Stress). This cycle focuses on reliability ("Self-Healing" from SCF failures) and efficiency ("Periodic Embedding" to cut out local structures).

**Key Features**:
*   **ASE Interface**: Wrapper around `ase.calculators.espresso` (Quantum Espresso) and generic VASP support.
*   **Self-Healing**: Automatically retry failed calculations by adjusting mixing parameters or smearing.
*   **Periodic Embedding**: Extract local clusters from large MD snapshots and embed them into small supercells for efficient DFT.

## 2. System Architecture

Files to be implemented/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── **oracle.py**           # Oracle Config & Results
│   └── ...
├── oracle/
│   ├── **__init__.py**
│   ├── **base.py**             # Abstract Base Class
│   ├── **manager.py**          # Self-Healing Logic
│   ├── **embedding.py**        # Periodic Embedding Logic
│   └── dft/
│       ├── **__init__.py**
│       ├── **espresso.py**     # QE Driver
│       └── **vasp.py**         # VASP Driver (Optional)
└── tests/
    └── **test_oracle/**
        ├── **test_manager.py**
        └── **test_embedding.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`src/mlip_autopipec/domain_models/oracle.py`)

*   **`CalculationResult`**:
    *   Fields: `energy` (float), `forces` (Nx3 Array), `stress` (Voigt 6 Array), `virial` (Optional), `success` (bool), `error_msg` (str), `walltime` (float).
*   **`DFTConfig`**:
    *   `code`: `QE` or `VASP`.
    *   `command`: `pw.x -in input > output` (string template).
    *   `kpoints`: `KpointsConfig` (grid or density).
    *   `pseudo_dir`: Path to pseudopotentials.

### 3.2. Oracle Component (`src/mlip_autopipec/oracle/`)

#### `base.py`
*   **`BaseOracle`** (ABC):
    *   `compute(structure: Structure) -> CalculationResult`
    *   Must handle timeouts and file cleanup.

#### `manager.py` (Self-Healing Logic)
*   **`OracleManager`**:
    *   Delegates to `BaseOracle`.
    *   **Retry Strategy**:
        *   If `SCF Convergence Error`:
            *   Attempt 1: Reduce mixing beta (e.g., 0.7 -> 0.3).
            *   Attempt 2: Increase smearing width (electron temperature).
            *   Attempt 3: Switch diagonalization algorithm (david -> cg).
        *   If `Walltime Exceeded`: Skip or resubmit to a larger queue (advanced).

#### `embedding.py` (Periodic Embedding)
*   **`ClusterEmbedder`**:
    *   **Input**: Large MD Structure + Target Atom Indices (high uncertainty).
    *   **Logic**:
        *   Extract atoms within $R_{cut} + R_{buffer}$ radius.
        *   Place in a box with vacuum? **NO**.
        *   Place in a small periodic supercell that fits the cluster.
        *   **Constraint**: Minimum image convention must be satisfied.
    *   **Output**: Small `Structure` ready for DFT.

#### `dft/espresso.py`
*   **`EspressoDriver`**:
    *   Uses `ase.calculators.espresso.Espresso`.
    *   Sets `tprnfor=True`, `tstress=True` always.
    *   Parses output for errors (using `ase` or custom regex).

## 4. Implementation Approach

1.  **Define Oracle Interfaces**: Create `base.py`.
2.  **Implement Embedding Logic**: Critical for efficiency. Use `ase.neighborlist`.
3.  **Implement Espresso Driver**: With basic input file generation.
4.  **Implement Self-Healing Manager**: A simple retry loop with modified calculator parameters.
5.  **Mock DFT**: For testing without installing QE.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_embedding.py`**:
    *   Create a large supercell with a known defect.
    *   Verify `ClusterEmbedder` extracts the correct atoms around the defect.
    *   Verify the resulting small cell has valid periodic boundary conditions.
*   **`test_manager.py`**:
    *   Mock `BaseOracle.compute` to raise `SCFError` on the first call.
    *   Assert `OracleManager` calls it a second time with modified parameters.

### 5.2. Integration Testing
*   **Real/Mock DFT**: If QE is installed (`shutil.which('pw.x')`), run a tiny Si calculation. Otherwise, use a Mock Calculator that returns LJ energies but via the `BaseOracle` interface.

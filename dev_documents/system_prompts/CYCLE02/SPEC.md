# Cycle 02 Specification: Oracle & Structure Generator

## 1. Summary

In Cycle 02, we replace the "Mocks" with real physics engines for the first half of the pipeline. The focus is on two critical capabilities: **Structure Generation** and **Ground Truth Labelling (Oracle)**.

We will implement the **Structure Generator**, capable of creating diverse atomic configurations. Initially, we will implement "Random Perturbation" (rattling atoms) and "Lattice Enumeration" (generating random supercells of bulk crystals). This replaces the `MockStructureGenerator`.

Simultaneously, we will implement the **Oracle** interface using the **ASE (Atomic Simulation Environment)** library. The first concrete implementation will be for **Quantum Espresso (QE)**. This module will handle the complex task of generating QE input files (`pw.x`), parsing the output (XML/Text), and extracting the Energy, Forces, and Stress (EFS) data. Crucially, we will implement a "Self-Healing" mechanism that can detect common DFT failures (like SCF non-convergence) and automatically retry with safer parameters (e.g., lower mixing beta).

We will also implement the **Periodic Embedding** logic. This is a sophisticated geometric algorithm that takes a local cluster of atoms (e.g., a defect site detected by the Dynamics Engine) and "wraps" it into a periodic supercell suitable for DFT calculation, ensuring that the local environment is preserved while satisfying periodic boundary conditions.

## 2. System Architecture

Files to be created/modified are marked in **bold**.

```
PYACEMAKER/
├── src/
│   └── mlip_autopipec/
│       ├── **structure_generator/**
│       │   ├── **__init__.py**
│       │   ├── **random_perturbation.py**  # Implements BaseStructureGenerator
│       │   └── **lattice_enumerator.py**
│       ├── **oracle/**
│       │   ├── **__init__.py**
│       │   ├── **qe_oracle.py**            # Implements BaseOracle via ASE
│       │   └── **embedding.py**            # Periodic Embedding Logic
│       ├── **domain_models/**
│       │   └── **structure.py**            # Enhanced with ASE atoms
│       └── config/
│           └── base_config.py              # Updated with QE config models
└── tests/
    ├── **unit/**
    │   ├── **test_structure_generator.py**
    │   └── **test_embedding.py**
    └── **integration/**
        └── **test_qe_oracle.py**
```

## 3. Design Architecture

### 3.1. Structure Generator Design
The `StructureGenerator` follows the Strategy Pattern.
*   **Input**: A seed structure (e.g., bulk Si) and a configuration (perturbation amount, supercell size).
*   **Output**: A list of `ase.Atoms` objects.
*   **Logic**:
    *   `RandomPerturbation`: Displaces each atom by a vector sampled from a Gaussian distribution.
    *   `LatticeEnumerator`: Generates supercells with different integer transformation matrices to sample volume/shape changes.

### 3.2. Oracle Design (The ASE Wrapper)
The `QEOracle` class wraps `ase.calculators.espresso.Espresso`.
*   **Self-Correction Loop**:
    ```python
    def compute(structure):
        try:
            return run_dft(structure, params)
        except ConvergenceError:
            params['mixing_beta'] *= 0.5  # Reduce mixing
            return run_dft(structure, params)
    ```
*   **Periodic Embedding**:
    *   **Problem**: We have a cluster of atoms cut from a large MD simulation. We need to calculate its forces.
    *   **Solution**: Create a new orthorhombic cell large enough to hold the cluster + buffer. Place the cluster in the centre. Apply Vacuum padding. This allows the plane-wave code to run without artifacts from image interactions (if vacuum is large) or allows us to simulate "bulk-like" conditions if we carefully manage the boundary.

## 4. Implementation Approach

1.  **Dependencies**: Add `ase` to `pyproject.toml`.
2.  **Domain Models**: Upgrade `Structure` to wrap or inherit from `ase.Atoms`. This gives us powerful geometry manipulation methods immediately.
3.  **Structure Generator**: Implement `random_perturbation.py`. Write a test that checks if atoms actually moved.
4.  **Embedding Logic**: Implement the geometric logic to create a box around a set of points. This is pure math/geometry.
5.  **QE Oracle**:
    *   Step A: Implement the basic `run` method using `ASE`.
    *   Step B: Implement the `SelfHealing` logic. Catch `ase.io.esp.EspressoError`.
    *   Step C: Create a test that uses a "Mock Calculator" in ASE to simulate a failure and verify the retry logic works.
6.  **Integration**: Update `GlobalConfig` to allow selecting `type: quantum_espresso`.

## 5. Test Strategy

### 5.1. Unit Testing Approach
*   **Structure Generator**: Verify that `perturb_structure(atoms, distance=0.1)` results in positions that are distinct from the original but within the expected delta. Verify that the cell parameters remain unchanged (or changed if intended).
*   **Embedding**: Create a synthetic cluster of 2 atoms. Verify that `embed_cluster` returns a Unit Cell of the specified size (e.g., 10x10x10) with the atoms centreed. Check that minimum image convention logic is correct.
*   **Config**: Verify that QE-specific parameters (pseudopotentials path, k-points) are correctly validated.

### 5.2. Integration Testing Approach
*   **Mock DFT Test**: We cannot rely on having `pw.x` installed in the CI environment. Therefore, we will create a `MockEspresso` calculator in Python that mimics the API of the real ASE Espresso calculator but runs instantly. We will use this to test the *orchestration* of the Oracle (input generation -> "run" -> output parsing).
*   **Real DFT Test (Local Only)**: We will provide a test marked `@pytest.mark.skipif(shutil.which('pw.x') is None)` that runs a real calculation on a tiny system (H2 molecule) to verify the actual binary interface works when available.

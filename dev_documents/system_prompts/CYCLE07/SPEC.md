# Cycle 07 Specification: Advanced Exploration (Structure Generator)

## 1. Summary

Cycle 07 upgrades the "Structure Generator" module. While Cycle 04/05 focused on MD exploration, this cycle introduces **Targeted Structural Engineering**. Instead of waiting for rare events to happen in MD (which might take nanoseconds or microseconds), we proactively generate structures that we *know* are physically important but hard to sample: defects (vacancies, interstitials) and strained lattices.

We also implement the **Adaptive Exploration Policy**. This logic engine decides *what* to do next based on the material's properties (e.g., if it's a metal, focus on melting; if it's a hard ceramic, focus on shear strain).

## 2. System Architecture

We populate the `generator/` directory.

### 2.1 File Structure

```ascii
src/mlip_autopipec/
├── generator/
│   ├── **__init__.py**
│   ├── **defects.py**              # Defect generation
│   ├── **distortions.py**          # Strain/Shear generation
│   └── **policy.py**               # Adaptive Strategy Logic
└── config/
    └── schemas/
        └── **generator.py**        # Defect config
```

## 3. Design Architecture

### 3.1 Defect Strategy (`generator/defects.py`)

*   **Responsibility**: Create supercells with point defects.
*   **Methods**:
    *   `generate_vacancy(atoms, supercell_matrix) -> List[Atoms]`: Uses symmetry (`spglib`) to generate only unique vacancies.
    *   `generate_interstitial(atoms, element) -> List[Atoms]`: Inserts atoms into Voronoi nodes or tetrahedral/octahedral sites.

### 3.2 Distortion Strategy (`generator/distortions.py`)

*   **Responsibility**: Apply affine transformations.
*   **Methods**:
    *   `apply_rattling(atoms, stdev)`: Randomly displace all atoms.
    *   `apply_strain(atoms, strain_tensor)`: Deform the box and atoms.

### 3.3 Adaptive Policy (`generator/policy.py`)

*   **Responsibility**: Determine the next set of tasks for the Dynamics Engine or Oracle.
*   **Logic**:
    *   Input: Current `WorkflowState` (e.g., current RMSE, material tags like "metal").
    *   Output: `ExplorationTask` (e.g., "Run NPT at 2000K" or "Generate 50 vacancies").

## 4. Implementation Approach

1.  **Step 1: Defect Generation.**
    *   Integrate `spglib` to find equivalent sites.
    *   Implement vacancy removal and interstitial insertion logic.

2.  **Step 2: Policy Engine.**
    *   Implement a rule-based system (Decision Tree).
    *   Example Rule: "If cycle < 3, focus on EOS (Strain). If cycle > 3, focus on Defects."

3.  **Step 3: Integration.**
    *   Connect this to the Orchestrator. The Orchestrator calls `StructureGenerator` at the start of a cycle to seed the pool of structures.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Symmetry Checks:**
    *   For a BCC crystal, asserting that generating vacancies yields exactly 1 unique structure (all sites are equivalent).
    *   For a compound like TiO2, asserting distinct vacancy sites are found.
*   **Policy Logic:**
    *   Test that the policy recommends high temperatures if `melting_point` is high.

### 5.2 Integration Testing
*   **Workflow Integration:**
    *   Run a cycle where the generator creates 10 defective structures.
    *   Verify these structures are passed to the Oracle (mocked) and result in new training data.

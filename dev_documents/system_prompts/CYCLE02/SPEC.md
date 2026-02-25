# Cycle 02 Specification: DIRECT Sampling & Active Learning

## 1. Summary
Cycle 02 focuses on implementing the first two steps of the 7-Step Distillation Workflow: **DIRECT Sampling** (Step 1) and **Uncertainty-based Active Learning** (Step 2).

The goal is to intelligently explore the chemical space rather than relying on random sampling. We will implement a `StructureGenerator` that uses the DIRECT (DIviding RECTangles) algorithm or a simplified "Entropy Maximization" strategy to generate a diverse initial pool of structures.

Simultaneously, we will enhance the `MaceSurrogateOracle` to compute **uncertainty** (variance or ensemble disagreement) alongside energy and forces. This uncertainty metric will be used by the `ActiveLearner` module to filter the generated structures, selecting only the most "informative" ones for DFT calculation.

Finally, we will introduce a `DFTOracle` (initially a mock) to simulate the ground-truth evaluation of these selected structures.

## 2. System Architecture

The following file structure will be created or modified. **Bold** files are new or significantly modified.

```text
src/pyacemaker/
├── generator/
│   ├── **__init__.py**
│   └── **direct.py**         # DIRECT / Entropy Maximization Implementation
├── modules/
│   ├── **__init__.py**
│   ├── **active_learner.py** # Selection Logic (Top-k Uncertainty)
│   ├── **oracle.py**         # Updated: DFTOracle and MaceSurrogateOracle
│   └── **structure_generator.py** # Updated: Uses DirectGenerator
├── domain_models/
│   └── **metrics.py**        # Uncertainty & Diversity Metrics
└── orchestrator.py           # Updated: Step 1 & 2 Workflow
```

## 3. Design Architecture

### 3.1. Structure Generator (`generator/direct.py`)
-   **`DirectGenerator`**: Implements `StructureGenerator` interface.
    -   **Algorithm**: Generates random structures, computes their descriptors (e.g., SOAP or simple pair distances), and uses a greedy algorithm (e.g., MaxMin distance) to select a subset that maximizes the coverage of the descriptor space.
    -   **Output**: A list of `StructureMetadata` objects with `generation_method="direct"`.

### 3.2. Uncertainty Quantification (`modules/oracle.py`)
-   **`MaceSurrogateOracle.compute_uncertainty`**:
    -   Computes the variance of predictions (ensemble or latent distance).
    -   Populates `structure.uncertainty_state` (`gamma_mean`, `gamma_max`).
    -   Must call `validate_structure_integrity` before processing.
    -   Must use efficient batch processing.

### 3.3. Active Learner (`modules/active_learner.py`)
-   **`ActiveLearner`**:
    -   **Input**: A pool of candidate structures with uncertainty scores.
    -   **Logic**: Selects the top $N$ structures with the highest `uncertainty_state.gamma_max`.
    -   **Optimization**: MUST use `heapq.nlargest` for O(K) memory usage (Streaming Selection).
    -   **Output**: A subset of `StructureMetadata` marked for DFT.

### 3.4. DFT Oracle (`modules/oracle.py`)
-   **`DFTOracle`**: Implements `Oracle`.
    -   **Real Mode**: Wraps ASE's VASP/QuantumEspresso calculator via `DFTManager`.
    -   **Mock Mode**: Returns a "True" energy using a simple pair potential (e.g., Lennard-Jones) plus some noise, ensuring deterministic but different values from MACE.

## 4. Implementation Approach

1.  **Implement Generator**: Create `DirectGenerator` in `generator/direct.py`. Implement a simple MaxMin diversity selection.
2.  **Enhance MACE Oracle**: Update `MaceSurrogateOracle` in `modules/oracle.py`. Add `compute_uncertainty`.
3.  **Implement DFT Oracle**: Ensure `DFTOracle` and `MockOracle` in `modules/oracle.py` behave correctly.
4.  **Implement Active Learner**: Create `ActiveLearner` in `modules/active_learner.py`. Implement `select_batch(candidates, n_select)`.
5.  **Update Orchestrator**:
    -   Update `_step1_direct_sampling`: Call `DirectGenerator`.
    -   Update `_step2_active_learning_loop`: Call `ActiveLearner` -> `Oracle`.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Generator Diversity**: Verify that `DirectGenerator` produces structures that are more diverse (higher mean pairwise distance in descriptor space) than random sampling.
-   **Active Learner**: Verify that `ActiveLearner.select_batch` correctly picks the items with the highest `gamma_max`. Verify memory usage with large mock iterator.
-   **Oracle**: Verify `validate_structure_integrity` is called.

### 5.2. Integration Testing
-   **Step 1-2 Flow**:
    1.  Initialize Orchestrator.
    2.  Run Step 1 (Generate 100 structures).
    3.  Run Step 2 (Select 10).
    4.  Verify that the Orchestrator holds a "DFT Dataset" with 10 labeled structures.
    5.  Verify that the labels (Energy/Forces) in the dataset come from the `DFTOracle` (check `label_source="dft"`).

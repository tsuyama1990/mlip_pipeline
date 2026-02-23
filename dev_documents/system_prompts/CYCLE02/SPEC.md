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
│   ├── **base.py**           # BaseGenerator Interface
│   └── **direct.py**         # DIRECT / Entropy Maximization Implementation
├── oracle/
│   ├── **dft.py**            # DFT Oracle (VASP/Mock Wrapper)
│   └── mace_oracle.py        # Added: uncertainty method
├── modules/
│   ├── **__init__.py**
│   └── **active_learner.py** # Selection Logic (Top-k Uncertainty)
├── domain_models/
│   └── **metrics.py**        # Uncertainty & Diversity Metrics
└── orchestrator.py           # Added: Step 1 & 2 Workflow
```

## 3. Design Architecture

### 3.1. Structure Generator (`generator/direct.py`)
-   **`DirectGenerator`**: Implements `BaseGenerator`.
    -   **Algorithm**: Generates random structures, computes their descriptors (e.g., SOAP or simple pair distances), and uses a greedy algorithm (e.g., MaxMin distance) to select a subset that maximizes the coverage of the descriptor space.
    -   **Output**: A list of `StructureData` objects with `source="DIRECT"`.

### 3.2. Uncertainty Quantification (`oracle/mace_oracle.py`)
-   **`MaceSurrogateOracle.predict_with_uncertainty`**:
    -   If using an ensemble MACE model, computes the variance of predictions.
    -   If using a single model, we might use a heuristic (e.g., distance in latent space) or `mace-torch`'s built-in uncertainty if available.
    -   **Return**: `StructureData` with populated `.uncertainty` field.

### 3.3. Active Learner (`modules/active_learner.py`)
-   **`ActiveLearner`**:
    -   **Input**: A pool of candidate structures with uncertainty scores.
    -   **Logic**: Selects the top $N$ structures with the highest uncertainty (or using a hybrid strategy like BatchBALD).
    -   **Output**: A subset of `StructureData` marked for DFT.

### 3.4. DFT Oracle (`oracle/dft.py`)
-   **`DFTOracle`**: Implements `BaseOracle`.
    -   **Real Mode**: Wraps ASE's VASP/QuantumEspresso calculator.
    -   **Mock Mode**: Returns a "True" energy using a simple pair potential (e.g., Lennard-Jones) plus some noise, ensuring deterministic but different values from MACE. This allows verifying that the system learns to correct errors.

## 4. Implementation Approach

1.  **Implement Generator**: Create `DirectGenerator` in `generator/direct.py`. Implement a simple MaxMin diversity selection.
2.  **Enhance MACE Oracle**: Add `predict_with_uncertainty` to `MaceSurrogateOracle`. For the mock, just return random values.
3.  **Implement DFT Oracle**: Create `DFTOracle` in `oracle/dft.py`. Implement the Mock behavior (LJ potential).
4.  **Implement Active Learner**: Create `ActiveLearner` in `modules/active_learner.py`. Implement `select_batch(candidates, n_select)`.
5.  **Update Orchestrator**:
    -   Add `run_step1_direct_sampling()`: Calls `DirectGenerator`.
    -   Add `run_step2_active_learning()`: Calls `MaceSurrogateOracle` -> `ActiveLearner` -> `DFTOracle`.
    -   Store the resulting "labeled" dataset in memory or a simple pickle file.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Generator Diversity**: Verify that `DirectGenerator` produces structures that are more diverse (higher mean pairwise distance in descriptor space) than random sampling.
-   **Active Learner**: Verify that `ActiveLearner.select_batch` correctly picks the items with the highest `.uncertainty` attribute.

### 5.2. Integration Testing
-   **Step 1-2 Flow**:
    1.  Initialize Orchestrator.
    2.  Run Step 1 (Generate 100 structures).
    3.  Run Step 2 (Select 10).
    4.  Verify that the Orchestrator holds a "DFT Dataset" with 10 labeled structures.
    5.  Verify that the labels (Energy/Forces) in the dataset come from the `DFTOracle` (check `source="DFT"`).

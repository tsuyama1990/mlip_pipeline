# Cycle 02 Specification: Data Management & Structure Generation

## 1. Summary

In Cycle 02, we breathe life into the "Generator" component and establish the data persistence layer. We replace the dummy mock generator with a functional **Structure Generator** capable of creating valid atomic structures (e.g., random alloys, surfaces). We also implement the **Dataset** class, which acts as the system's memory, ensuring that every structure calculated by the Oracle is safely stored and easily retrievable for training. Finally, we introduce the **Adaptive Exploration Policy** engine, which will eventually guide the search, although in this cycle we will implement a basic version that selects strategies based on simple configuration rules.

## 2. System Architecture

Files in **bold** are the focus of this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── **random.py**       # Random structure generator
│   │   └── **adaptive.py**     # Adaptive Policy Logic
├── core/
│   ├── **dataset.py**          # Real Dataset implementation
└── domain_models/
    ├── **structure.py**        # (Enhance with validation)
```

## 3. Design Architecture

### 3.1. Dataset (`core/dataset.py`)
The `Dataset` class is responsible for O(1) appending and efficient loading of training data.
-   **Storage Format**: `JSONL` (JSON Lines) is chosen for human readability and append-only safety, or `Pickle` for performance with large numpy arrays. We will use a hybrid approach or standard ASE formats (`.extxyz`).
-   **Schema**:
    -   `structures`: List[Structure]
    -   `metadata`: Dict (stats, timestamp)
-   **Methods**:
    -   `append(structures: list[Structure])`: Writes to disk immediately.
    -   `load() -> list[Structure]`: Reads from disk.
    -   `__len__()`: Returns count.

### 3.2. Structure Generator (`components/generator/`)
We implement concrete classes inheriting from `BaseGenerator`.
-   **`RandomGenerator`**:
    -   Input: `composition` (e.g., {"Fe": 0.5, "Pt": 0.5}), `supercell_size`.
    -   Logic: Creates a lattice (fcc/bcc) and randomly populates sites. Randomly perturbs positions ("rattling").
    -   Validation: Ensures atoms are not too close (min_distance check).

### 3.3. Adaptive Policy (`components/generator/adaptive.py`)
A decision-making module used by the Generator.
-   **Input**: Current Cycle ID, Previous uncertainty metrics.
-   **Output**: A `GenerationStrategy` (e.g., "Run MD at 300K", "Generate Random Alloy").
-   **Logic (Cycle 02)**: Simple heuristic (e.g., "If Cycle < 2, use Random; else use MD").

## 4. Implementation Approach

1.  **Enhance Structure Model**: Add validators to `domain_models/structure.py` to ensure `positions` and `atomic_numbers` match in length.
2.  **Implement Dataset**: Create `core/dataset.py`. Use `ase.io.write` (append mode) or custom JSONL writer. Ensure it handles the `Structure` Pydantic model.
3.  **Implement RandomGenerator**: Use `ase.build.bulk` to create base lattices and `numpy.random` to substitute atoms.
4.  **Integrate**: Update `factory.py` to allow `type: random`.
5.  **Test**: Run the pipeline with `type: random` and verify `dataset.jsonl` (or `.extxyz`) is created and populated.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Dataset**:
    -   Test appending 10 structures.
    -   Test loading them back. Verify data integrity (positions match).
-   **RandomGenerator**:
    -   Test generation with specific composition (Fe50Pt50).
    -   Verify the output `Structure` has the correct number of atoms and species.
    -   Verify minimum distance constraint (no overlapping atoms).

### 5.2. Integration Testing
-   **Scenario**: "Data Accumulation"
-   **Config**: Generator=`random`, Oracle=`mock` (returns structures with fake energy), Trainer=`mock`.
-   **Check**: Run 3 cycles. The dataset file should contain $3 \times N_{gen}$ structures.

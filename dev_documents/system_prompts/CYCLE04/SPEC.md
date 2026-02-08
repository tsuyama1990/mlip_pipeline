# Cycle 04: Trainer (Pacemaker Integration)

## 1. Summary

In this cycle, we implement the **Trainer**, which serves as the bridge between our labeled dataset and the `pacemaker` library. The Trainer is responsible for fitting the Atomic Cluster Expansion (ACE) potential to the Energy, Force, and Stress data provided by the Oracle.

To maximize data efficiency, we integrate **Active Set Selection** (using `pace_activeset`). Instead of training on every single structure ever generated—which leads to redundancy and slow training—we use D-optimality criteria to select only the most "informative" structures (those that maximize the determinant of the information matrix).

Crucially, we also implement **Delta Learning**. The Trainer will be configured to learn the *difference* between the DFT energy and a physical baseline (Lennard-Jones or ZBL). This ensures that the final potential inherits the robust short-range repulsion of the baseline, preventing unphysical behavior in the absence of data.

## 2. System Architecture

Files in **bold** are new or modified.

```ascii
src/mlip_autopipec/
├── components/
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── **pacemaker.py**      # Pacemaker Wrapper
│   │   └── **activeset.py**      # Active Set Logic
│   └── ...
```

## 3. Design Architecture

### 3.1. Pacemaker Wrapper (`pacemaker.py`)
-   **Class `PacemakerTrainer`**:
    -   `train(dataset: Dataset, prev_potential: Potential = None) -> Potential`
    -   **Steps**:
        1.  **Prepare Data**: Convert `Dataset` (JSONL/pckl) to Pacemaker's expected format (`input.pckl.gzip`).
        2.  **Select Active Set**: Run `pace_activeset` to filter the data.
        3.  **Configure**: Generate `input.yaml` for `pace_train`.
            -   Set `fitting: {weight_energy: 1.0, weight_force: 1.0}`.
            -   Set `backend: {evaluator: tensorpot}`.
        4.  **Run Training**: Execute `pace_train`.
        5.  **Cleanup**: Collect `potential.yace` and logs.

### 3.2. Active Set Logic (`activeset.py`)
-   **Class `ActiveSetSelector`**:
    -   `select(dataset: Dataset, max_structures: int = 1000) -> Dataset`
    -   Wraps `pace_activeset` command.
    -   **Algorithm**: MaxVol.

### 3.3. Delta Learning Configuration
-   The `GlobalConfig` will have a section `physics_baseline`:
    -   `type`: "lj" or "zbl"
    -   `params`: `{epsilon: 0.1, sigma: 2.5}`
-   The Trainer must read this and inject it into the `input.yaml` as a reference potential.

## 4. Implementation Approach

1.  **Refactor**: Ensure `Dataset` can export to `pandas` DataFrame or directly to `pckl.gzip` (Pacemaker format).
2.  **Active Set**: Implement `ActiveSetSelector`.
    -   Use `subprocess.run(["pace_activeset", ...])`.
    -   Parse the output to identify selected structures.
3.  **Trainer**: Implement `PacemakerTrainer`.
    -   Template the `input.yaml` using `jinja2` or `yaml.dump`.
    -   Handle `prev_potential` for fine-tuning (start from existing weights).
4.  **Integration**: Update `Orchestrator` to call `Trainer.train()`.

## 5. Test Strategy

### 5.1. Unit Tests
-   **Config Generation**: verify `input.yaml` contains correct weights, cutoffs, and baseline settings.
-   **Command Construction**: verify `pace_train` command string includes all flags (`--initial_potential`, etc.).

### 5.2. Integration Tests (Mock Pacemaker)
-   **Mock Binary**: Create a dummy `pace_train` script that:
    -   Reads `input.yaml`.
    -   Writes a dummy `potential.yace`.
    -   Writes a `log.txt` with "Training Finished".
-   **Run**: `Trainer.train(dataset)`. Verify artifacts are created.

### 5.3. Real Training Test (Requires Pacemaker)
-   **Tiny Dataset**: 10 structures of LJ Argon.
-   **Run**: Train a simple potential (1 layer, small basis).
-   **Validation**:
    -   `potential.yace` is created.
    -   The RMSE on training set is small (< 0.1 eV/A).

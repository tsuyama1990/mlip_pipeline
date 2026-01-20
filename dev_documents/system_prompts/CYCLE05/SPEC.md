# Cycle 05: Inference & Scalable OTF

## 1. Summary

Cycle 05 represents the closing of the feedback loop. In Cycles 1-4, we built a factory to create a Machine Learning Potential (MLP). In Cycle 05, we implement **Module E: Scalable Inference & On-The-Fly (OTF)**. This module uses the trained MLP to perform actual materials science simulations (Molecular Dynamics), exploring the phase space to find new physics and—crucially—new failure modes.

An MLP is an interpolator. It is extremely accurate within the domain of its training data but unreliable outside of it. When a simulation explores new phase space (e.g., a solid melting into a liquid, or a crack propagating), the model's predictions may deviate from reality. To handle this, we implement an **Active Learning** strategy based on **Uncertainty Quantification (UQ)**.

The Inference Engine runs massive LAMMPS simulations (millions of atoms). At every step (or interval), it monitors the **Extrapolation Grade ($\gamma$)** of the system. This metric, provided by the ACE potential, quantifies how "far" the current atomic environment is from the training set basis.
-   If $\gamma$ is low (< 2.0), the simulation proceeds.
-   If $\gamma$ exceeds a threshold (e.g., > 5.0), the simulation is paused, or the problematic frame is flagged.

When a high-uncertainty configuration is found, we face a scale mismatch: the simulation might have 100,000 atoms, but DFT scales as $O(N^3)$ and can only handle ~200 atoms. We cannot simply send the simulation snapshot to DFT. To solve this, we implement a **Periodic Embedding Strategy**. We identify the specific atom responsible for the high uncertainty, cut out a small cluster (radius $\sim 5\AA$) around it, and wrap this cluster in a new, small periodic box with a buffer zone. This "surgically extracted" sample captures the local physics of the failure mode but is small enough for fast DFT calculation.

By the end of this cycle, the system will be able to run long-time MD simulations, automatically detect when the model starts to "hallucinate," and extract precise training samples to fix the error, enabling the creation of self-healing potentials.

## 2. System Architecture

### 2.1. Code Blueprint and File Structure

The inference module manages the interface with LAMMPS and the complex geometry of embedding.

The following file structure will be implemented. Files in **bold** are the primary deliverables.

```
mlip_autopipec/
├── inference/
│   ├── **__init__.py**
│   ├── **config.py**               # Pydantic schemas for MD settings (Temp, Steps)
│   ├── **lammps_runner.py**        # Wrapper for running LAMMPS MD
│   ├── **uq.py**                   # Logic for analyzing uncertainty logs/outputs
│   ├── **embedding.py**            # Geometry logic for cluster extraction and PBC
│   └── **templates/**              # Jinja2 templates for LAMMPS scripts
│       └── **md.in.j2**            # The generic MD input script
└── tests/
    └── inference/
        ├── **test_lammps.py**
        ├── **test_embedding.py**
        └── **test_uq.py**
```

### 2.2. Component Interaction and Data Flow

1.  **Input**:
    The Orchestrator provides:
    -   Path to the latest `.yace` potential.
    -   An initial atomic structure (from Generator or previous run).
    -   An `InferenceConfig` (e.g., "Heat to 1000K").

2.  **Simulation Setup**:
    The `LammpsRunner` prepares the run directory.
    -   It writes the structure to `data.lammps`.
    -   It uses `jinja2` to render `md.in`. Key settings:
        -   `pair_style pace`: Activates the ML potential.
        -   `compute max_gamma all pace/extrapolation`: Computes the UQ metric.
        -   `fix stop_check all halt 100 v_max_gamma > 5.0`: Directs LAMMPS to abort if uncertainty is too high.

3.  **Execution**:
    `lmp_mpi` runs the simulation. The runner monitors the process.
    -   **Scenario A (Success)**: Simulation finishes. Trajectory is saved.
    -   **Scenario B (Uncertainty Stop)**: Simulation aborts early. The final snapshot is saved.

4.  **Analysis (The UQ Check)**:
    The `UncertaintyChecker` parses the log file.
    -   If the run stopped due to high gamma, it identifies the timestamp.
    -   It loads the final snapshot.

5.  **Extraction (The Embedding Logic)**:
    The `EmbeddingExtractor` takes the large snapshot.
    -   It finds the atom index $i$ with the highest contribution to the uncertainty.
    -   It cuts a spherical cluster of radius $R_{cut}$ around atom $i$.
    -   It places this cluster in a new cubic box of size $L \approx 2 R_{cut} + \delta$.
    -   It handles Periodic Boundary Conditions (PBC) to ensure the cluster is physically continuous.
    -   **Masking**: It generates a `force_mask` array. Atoms in the inner core ($r < R$) get weight 1.0. Atoms in the buffer shell ($R < r < R+\delta$) get weight 0.0.

6.  **Output**:
    The new small `Atoms` object is sent to the DB with `status=PENDING_DFT`.

## 3. Design Architecture

### 3.1. Inference Configuration (`inference/config.py`)

-   **`InferenceConfig`**:
    -   `temperature`: `float`. Target temperature in Kelvin.
    -   `pressure`: `float` (default 0.0). Target pressure in Bar.
    -   `steps`: `int` (default 10,000). Duration.
    -   `timestep`: `float` (default 0.001 ps).
    -   `uncertainty_threshold`: `float` (default 5.0). Gamma value to trigger stop.
    -   `sampling_interval`: `int` (default 100). How often to check UQ.

### 3.2. LAMMPS Runner (`inference/lammps_runner.py`)

-   **Class**: `LammpsRunner`
-   **Method**: `run_md(atoms, potential_path, config) -> SimulationResult`
    -   **Templating**: The Jinja2 template is crucial here. It must support NVT (canonical) and NPT (isobaric) ensembles.
    -   **Error Handling**: LAMMPS can crash with "Lost Atoms" if the potential is very bad. The runner catches this `subprocess.CalledProcessError`. It attempts to recover the *last valid frame* from the dump file and treat it as a high-uncertainty candidate (since bad forces caused the crash).

### 3.3. Embedding Extractor (`inference/embedding.py`)

This is the most mathematically complex component, dealing with Euclidean geometry under periodic conditions.

-   **Class**: `EmbeddingExtractor`
-   **Method**: `extract_cluster(supercell: Atoms, center_index: int, radius: float, buffer: float) -> Atoms`
    -   **Algorithm**:
        1.  **Center**: Translate the entire system so `center_index` is at $(0,0,0)$. Apply PBC wrap.
        2.  **Select**: Use KD-Tree or NeighborList to find all atoms within distance $R_{total} = radius + buffer$.
        3.  **Box**: Define a new cubic cell of side $L = 2 R_{total} + \epsilon$.
        4.  **Transfer**: Copy the selected atoms to a new `Atoms` object with the new cell.
        5.  **Mask**: Iterate through the new atoms. Calculate distance to center.
            -   If $d < radius$: `force_mask[i] = 1.0`.
            -   If $radius \le d < R_{total}$: `force_mask[i] = 0.0`.
    -   **Validation**: Check minimum distance between any pair of atoms in the new box (including periodic images). If overlap < 0.5A, the box is too small.

### 3.4. Uncertainty Checker (`inference/uq.py`)

-   **Class**: `UncertaintyChecker`
-   **Method**: `analyze_log(log_path) -> UQResult`
    -   **Parsing**: Reads LAMMPS log line by line. Looks for custom output `v_max_gamma`.
    -   **Logic**: Returns the timestep where `max_gamma` first exceeded the threshold.

## 4. Implementation Approach

1.  **Phase 1: Geometry (The Hard Part)**
    -   Implement `EmbeddingExtractor`. This relies purely on `ase` and `numpy`.
    -   Test heavily with unit tests. Create a 1D chain `A-B-C-D-E` with PBC. Extract `C`. Verify we get `B-C-D` in a new box.
    -   Verify force masks are correct.

2.  **Phase 2: Templates**
    -   Create `md.in.j2`. Ensure it works for both `units metal` (standard for materials) and `units real` (if needed).
    -   Verify syntax with a dry run if possible.

3.  **Phase 3: LAMMPS Integration**
    -   Implement `LammpsRunner`.
    -   **Dependency**: We assume `lmp_mpi` is in the path.
    -   **Mock**: For CI, we mock `subprocess.run`. The mock reads the input script and writes a dummy trajectory and log file. The log file can be configured to simulate a "high gamma" event at step 500.

4.  **Phase 4: Connection**
    -   Write an integration script that takes a structure -> runs (mock) MD -> detects stop -> extracts cluster -> saves to XYZ.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Embedding Logic**:
    -   **Test Case**: 3x3x3 Supercell of Aluminum. Pick an atom. Extract radius $4\AA$.
    -   **Assert**: Resulting system has ~50 atoms.
    -   **Assert**: Central atom is at the center of the new box.
    -   **Assert**: Boundary atoms have `mask=0`.
-   **Config Validation**:
    -   Test that `temperature < 0` raises `ValidationError`.

### 5.2. Integration Testing
-   **Mock MD Run**:
    -   Setup: Configure runner to run 1000 steps.
    -   Mock Behavior: The mock binary writes a log file where `max_gamma` jumps to 10.0 at step 500.
    -   Action: `runner.run()`.
    -   Assert: Runner detects the stop. Returns `SimulationResult` with `status=UNCERTAIN` and `trigger_step=500`.
-   **Real MD Run (Local)**:
    -   Requires `lammps` with `PACE` package.
    -   Run a short simulation on Al.
    -   Verify energy conservation (NVE ensemble).

### 5.3. Corner Cases
-   **Explosion**: If atoms fly apart, the embedding extractor might find 0 neighbors. Code should handle empty cluster gracefully (raise Error).
-   **Box Size**: If extracted box is smaller than cutoff, atoms interact with themselves. Extractor must enforce `box_size > 2 * cutoff`.

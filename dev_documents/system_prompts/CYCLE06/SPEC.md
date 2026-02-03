# Cycle 06 Specification: Advanced Physics & Deployment

## 1. Summary

Cycle 06 is the final polish. It transforms the system from a "Potential Generator" into a "Scientific Discovery Platform". This cycle adds the advanced physics capabilities required for the Grand Challenge: bridging the timescale gap using Adaptive Kinetic Monte Carlo (aKMC) via the EON software, and rigorously validating the potential's physical properties (Phonons, Elastic Constants). Finally, it implements the "Production Deployer" to package the results into a distributable artifact, and includes the development of the high-quality tutorial notebooks.

## 2. System Architecture

### 2.1 File Structure

**Bold** files are to be created or modified in this cycle.

```text
.
├── src/
│   └── mlip_autopipec/
│       ├── physics/
│       │   ├── dynamics/
│       │   │   ├── **eon_wrapper.py**  # EON Integration
│       │   │   └── **pace_driver.py**  # EON->Pacemaker bridge
│       │   └── validation/
│       │       ├── **metrics.py**      # Phonons/Elasticity logic
│       │       └── **report_generator.py** # HTML Report
│       ├── infrastructure/
│       │   └── **production.py**       # ProductionDeployer
│       └── **main.py**                 # Update CLI with 'package' command
├── tutorials/
│   ├── **01_quickstart_silicon.ipynb**
│   ├── **02_advanced_tio2.ipynb**
│   └── **04_grand_challenge_fept.ipynb**
└── tests/
    ├── unit/
    │   └── **test_production.py**
    └── integration/
        └── **test_eon_bridge.py**
```

## 3. Design Architecture

### 3.1 EON Bridge (`eon_wrapper.py` & `pace_driver.py`)
-   **Challenge**: EON is an external C++ code that expects a specific client-server protocol or a standalone executable to calculate Forces/Energies.
-   **Solution**: We implement `pace_driver.py`, a standalone script that reads atomic coordinates from `stdin` (or EON's .con format), invokes the `pyacemaker.calculator`, and prints the results to `stdout`.
-   **Integration**: The `EonWrapper` class manages the `eonclient` process, setting up the `config.ini` to point to our `pace_driver.py`.

### 3.2 Advanced Validation (`metrics.py`)
-   **Phonons**: Use `phonopy` (if installed) to calculate the band structure.
    -   *Check*: Iterate through q-points. If any frequency $\omega < -0.1$ THz (imaginary), flag as unstable.
-   **Elasticity**: Apply $\pm 1\%$ strains to the unit cell. Fit the Energy-Strain curve to extract $C_{11}, C_{12}, C_{44}$. Compare with experimental values provided in `config.yaml` (optional targets).

### 3.3 Production Deployer (`infrastructure/production.py`)
-   **Role**: Create the final artifact.
-   **Artifact Content**:
    -   `potential.yace`: The file.
    -   `manifest.json`: Training history, version, author, timestamp.
    -   `validation_report.html`: The visual report.
    -   `LICENSE`: Usage rights.
-   **Output**: A versioned ZIP file (e.g., `TiO2_v1.0.0.zip`).

## 4. Implementation Approach

1.  **Pace Driver**:
    -   Write `src/mlip_autopipec/physics/dynamics/pace_driver.py`. This script must be executable and independent, able to load the `.yace` file specified in its environment or arguments.

2.  **Validation Logic**:
    -   Implement `ElasticValidator` class.
    -   Implement `PhononValidator` class (wrapping `phonopy`).
    -   Implement `ReportGenerator` using `jinja2` to create HTML from the results.

3.  **Deployment**:
    -   Implement `ProductionDeployer.package(potential_path, output_path)`.

4.  **Tutorials**:
    -   Write the Jupyter Notebooks defined in `FINAL_UAT.md`.
    -   Ensure they use the "Mock Mode" flag to run quickly in CI.

## 5. Test Strategy

### 5.1 Unit Testing
-   **`test_production.py`**:
    -   Call `ProductionDeployer.package`.
    -   Verify the ZIP file exists and contains the manifest.
-   **`test_eon_bridge.py`**:
    -   Run `pace_driver.py` via subprocess, piping in a simple atomic configuration.
    -   Verify it outputs formatted Energy and Forces.

### 5.2 Integration Testing
-   **Scientific Verification**:
    -   Run the `01_quickstart_silicon.ipynb` notebook as a test.
    -   This exercises the full stack + validation + plotting.
    -   Assertion: The notebook execution completes with no exceptions.

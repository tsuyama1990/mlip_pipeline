# Cycle 06 UAT: The Final Exam

## 1. Test Scenarios

### 1.1. Scenario: Validation Report Generation
**ID**: UAT-CY06-001
**Priority**: High
**Description**: Verify that the system automatically generates a readable validation report after training.

**Steps:**
1.  **Setup**: Provide a trained `.yace` potential.
2.  **Execution**: Run `python -m mlip_autopipec.main validate potential.yace`.
3.  **Observation**: The system runs Elastic, Phonon, and EOS tasks.
4.  **Verification**: A file `validation_report.html` is created. Open it and check for:
    *   Table of Elastic Constants ($C_{11}, C_{12}, C_{44}$).
    *   Phonon Band Structure plot.
    *   "PASS/FAIL" badge.

### 1.2. Scenario: "Fe/Pt on MgO" Tutorial (End-to-End)
**ID**: UAT-CY06-002
**Priority**: Critical
**Description**: Execute the full user story defined in `USER_TEST_SCENARIO.md` using the "Mock" mode to ensure the workflow is unbroken.

**Steps:**
1.  **Setup**: `export CI=true`.
2.  **Execution**: Run the Jupyter Notebook `tutorials/01_MgO_FePt_Training.ipynb` using `pytest --nbval`.
3.  **Verification**:
    *   Notebook executes all cells without error.
    *   Outputs `potential.yace`.
4.  **Execution**: Run `tutorials/02_Deposition_and_Ordering.ipynb`.
5.  **Verification**:
    *   Outputs `deposition.dump`.
    *   Logs show successful mock-up of kMC ordering.

### 1.3. Scenario: Production Deployment
**ID**: UAT-CY06-003
**Priority**: Medium
**Description**: Verify that the potential is packaged correctly for use in external LAMMPS simulations.

**Steps:**
1.  **Setup**: A "Production Ready" potential.
2.  **Execution**: Run `main.py export --format=lammps_plugin`.
3.  **Verification**:
    *   A folder `deploy/` is created.
    *   It contains the `.yace` file and a `README.txt` with the exact `pair_style` and `pair_coeff` commands needed to use it.

## 2. Behaviour Definitions

**Feature**: Automated Quality Assurance

**Scenario**: Detecting Unstable Potentials

**GIVEN** a newly trained potential that overfits (predicts imaginary phonons)
**WHEN** the Validation Suite runs
**THEN** the Phonon test should detect $\omega^2 < -0.1$ THz
**AND** the test result should be "FAIL"
**AND** the system should NOT promote this potential to "Production"
**AND** the Orchestrator should flag the need for more data in that region

**Feature**: Documentation & Usability

**Scenario**: First Time User Experience

**GIVEN** a user with a fresh clone of the repo
**WHEN** they run `python main.py --help`
**THEN** they should see clear, grouped commands (run, validate, export)
**AND** when they run the tutorial notebook
**THEN** all imports should work (assuming `uv sync` was run)
**AND** the notebook should explain each step clearly

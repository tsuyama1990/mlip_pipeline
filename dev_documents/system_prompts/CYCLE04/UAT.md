# Cycle 04 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario ID: UAT-C04-001 - The First Quantum Leap (Running a DFT Calculation)

**Priority:** High
**Description:**
This is a critical milestone. We verify that the system can bridge the gap between the "Python World" (ASE atoms) and the "Fortran World" (Quantum Espresso). The user manually triggers a calculation for a specific structure and verifies that the results (Forces, Energy) are correctly stored in the database.

**User Story:**
As a Researcher, I want to manually process a specific candidate structure (e.g., ID=1) to verify my DFT settings (pseudopotentials, cutoffs) are correct. I expect the system to generate the input file, run `pw.x`, and automatically save the results without me parsing text files manually.

**Step-by-Step Walkthrough:**
1.  **Preparation**: The user ensures `pw.x` is in the path (or uses a mock). They have a database with 1 PENDING structure (e.g., Silicon).
2.  **Configuration**: The user sets `dft.command: mpirun -np 4 pw.x`.
3.  **Execution**: The user runs `mlip-auto run-job --id 1`.
    -   *Expectation*: The CLI logs: "Starting job 1...", "Generating inputs...", "Running QE...", "Parsing results...".
    -   *Expectation*: The process finishes with "Job 1 Completed Successfully."
4.  **Verification (Database)**: The user inspects the database.
    -   `status`: Changed from `Pending` to `Completed`.
    -   `energy`: Should be present (approx -1000 eV).
    -   `forces`: A (N,3) array should be stored.
    -   `stress`: A (3,3) array should be stored.
5.  **Verification (Files)**: The user checks the working directory.
    -   *Expectation*: The temporary folder is gone (cleanup successful).

**Success Criteria:**
-   The status transition is correct.
-   The numerical data is populated.
-   No zombie processes are left running.

### Scenario ID: UAT-C04-002 - Handling Invalid Pseudopotentials (Fail Fast)

**Priority:** Medium
**Description:**
A common error is missing pseudopotential files. The system should detect this before launching the MPI process.

**User Story:**
As a User, I accidentally pointed `pseudopotential_dir` to an empty folder. I want the system to tell me immediately that it can't find "Si.upf" rather than crashing with a cryptic Fortran error inside the MPI runtime.

**Step-by-Step Walkthrough:**
1.  **Configuration**: User sets `dft.pseudopotential_dir` to `/tmp/empty`.
2.  **Execution**: User runs `mlip-auto run-job --id 1`.
3.  **Result**: The CLI prints a generic error or specific error.
    -   *Expectation*: "Error: Pseudopotential for element 'Si' not found in /tmp/empty."
4.  **Database State**:
    -   *Expectation*: The job status remains `Pending` (or `Failed` with a clear reason), not stuck in `Running`.

**Success Criteria:**
-   Clear error message identifying the missing file.
-   Graceful exit.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Automated DFT Execution
  As a Compute Node
  I want to execute Quantum Espresso calculations based on database entries
  So that I can generate training data for the ML model

  Background:
    Given the database contains a structure with ID 100
    And the structure is status "PENDING"
    And the DFT configuration is valid

  Scenario: Successful SCF Calculation
    When I trigger the DFT runner for ID 100
    Then the input file "pw.in" should be generated
    And the command "pw.x" should be executed
    And the output should be parsed
    And the database entry 100 should have status "COMPLETED"
    And the database entry 100 should contain "energy" and "forces"

  Scenario: Automatic K-point Generation
    Given a large supercell (10x10x10 Angstrom)
    And a small primitive cell (3x3x3 Angstrom)
    And a k-spacing of 0.2 1/Angstrom
    When the input file is generated
    Then the large supercell should have fewer k-points (e.g., 2x2x2)
    And the small primitive cell should have more k-points (e.g., 6x6x6)
```

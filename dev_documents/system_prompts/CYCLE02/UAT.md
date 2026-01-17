# Cycle 02: User Acceptance Testing (UAT)

## 1. Test Scenarios

Cycle 02 is about trust. The user needs to trust that when the system says "Running DFT", it is doing so correctly and robustly. These tests verify the automation of the DFT Factory.

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-02-01** | High | Heuristic Parameter Generation | Verify that the system automatically assigns correct K-points and magnetic settings for a known magnetic material (e.g., BCC Iron). This ensures physical correctness without user input. |
| **UAT-02-02** | High | Auto-Recovery from Divergence | Simulate a DFT calculation that fails to converge initially. Verify the system detects the error, adjusts mixing parameters, and successfully completes the calculation. This is the core "Zero-Human" feature. |
| **UAT-02-03** | Medium | SSSP Pseudopotential Loading | Verify that the system correctly maps elements (e.g., Si, O) to their standard pseudopotential files and cutoff energies. |

### Recommended Notebooks
*   `notebooks/UAT_02_DFT_Factory.ipynb`:
    1.  Define an Fe crystal.
    2.  Instantiate `DFTRunner` (pointing to a Mock QE executable).
    3.  Run the job.
    4.  Inspect the generated input file to prove `nspin=2` was set.
    5.  Inspect the logs to see the "Retry" event triggered by the mock failure.

## 2. Behavior Definitions

### UAT-02-01: Heuristics

**Narrative**:
The user provides a structure of Iron (Fe). They do not know the optimal K-point density or the magnetic moment. They rely on the system to set these. The system analyzes the cell (lattice vectors) and the elements. It recognizes Iron as a magnetic element and sets the spin polarization flag. It calculates a dense grid for the small unit cell.

```gherkin
Feature: Heuristic Parameter Selection

  Scenario: Configuring a Magnetic Material (Iron)
    GIVEN a BCC Iron (Fe) crystal structure with lattice constant 2.87 A
    WHEN the Heuristic Engine processes the structure
    THEN "nspin" should be set to 2 (spin polarized)
    AND the initial magnetic moments for Fe atoms should be set to a non-zero value (e.g., 5.0)
    AND the K-point mesh density should be approximately 0.15 inverse Angstroms (resulting in roughly 14x14x14 grid)
    AND the pseudopotential for Fe should be the PBE-Sol version from SSSP

  Scenario: Configuring a Non-Magnetic Material (Silicon)
    GIVEN a Diamond Silicon (Si) crystal structure
    WHEN the Heuristic Engine processes the structure
    THEN "nspin" should be set to 1 (non-spin polarized)
    OR "nspin" should be absent (defaulting to 1)
```

### UAT-02-02: Auto-Recovery

**Narrative**:
A calculation for a difficult slab structure is running. The default mixing beta (0.7) is too aggressive, and the SCF cycle oscillates, failing to converge after 100 steps. The `DFTRunner` detects the "convergence NOT achieved" message in the output. It consults the `AutoRecovery` module, which prescribes reducing beta to 0.3. The Runner restarts the job with the new parameter. This time it converges.

```gherkin
Feature: Automated Error Recovery

  Scenario: Recovering from SCF Convergence Failure
    GIVEN a DFTRunner configured with a Mock QE executable
    AND the Mock QE is programmed to fail the first run with "convergence NOT achieved"
    AND the Mock QE is programmed to succeed on the second run
    WHEN the DFTRunner executes the calculation
    THEN the system should log a "Convergence Error" detection
    AND the system should retry the calculation automatically
    AND the second attempt should use a lower "mixing_beta" (e.g., 0.3) than the first (0.7)
    AND the final result should be a successful termination
    AND the returned Atoms object should contain Energy and Forces
    AND the history log should show 1 failure and 1 success

  Scenario: Unrecoverable Error
    GIVEN a DFTRunner configured with a Mock QE executable
    AND the Mock QE is programmed to fail with "Disk Quota Exceeded"
    WHEN the DFTRunner executes the calculation
    THEN the system should log an "Unknown Error"
    AND the system should NOT retry infinitely
    AND the system should raise a specialized DFTError exception
```

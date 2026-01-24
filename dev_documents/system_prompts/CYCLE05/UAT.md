# Cycle 05 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario ID: UAT-C05-001 - The Self-Healing Calculation (Convergence Recovery)

**Priority:** High
**Description:**
In high-throughput studies, roughly 5-10% of calculations fail due to default parameters being insufficient for complex electronic states (e.g., magnetic frustration). This test proves that the system can autonomously recover from a standard SCF convergence failure without human intervention.

**User Story:**
As a Project Manager, I want the system to automatically fix "convergence not achieved" errors by lowering the mixing beta, so that I don't have to manually check log files and resubmit jobs on the weekend.

**Step-by-Step Walkthrough:**
1.  **Preparation**: The user (or test harness) configures a mock environment where `pw.x` is replaced by a script that outputs a "convergence not achieved" error unless the input file contains `mixing_beta = 0.3`.
2.  **Execution**: The user submits a job for a difficult structure (e.g., ID 99).
3.  **Monitoring**: The user watches the logs (tail -f log.txt).
    -   *Log*: "Job 99 started."
    -   *Log*: "Job 99 failed: Convergence Error."
    -   *Log*: "Recovery Strategy Applied: mixing_beta -> 0.3"
    -   *Log*: "Retrying Job 99 (Attempt 2)..."
    -   *Log*: "Job 99 Completed Successfully."
4.  **Verification**:
    -   The database status is `COMPLETED`.
    -   The `data` column contains the result from the *second* run.
    -   A metadata field `retries` shows `1`.

**Success Criteria:**
-   The job does not stop at the first failure.
-   The correct parameter was changed.
-   The final result is stored.

### Scenario ID: UAT-C05-002 - The "Lost Cause" (giving up gracefully)

**Priority:** Medium
**Description:**
Some structures are just physically impossible (e.g., overlapping atoms that slipped through the surrogate). The recovery logic must not loop infinitely. It must give up after N tries.

**User Story:**
As an Administrator, I want the system to mark a job as FAILED after 5 unsuccessful attempts, so that it doesn't waste infinite compute cycles on a bad structure.

**Step-by-Step Walkthrough:**
1.  **Preparation**: The mock `pw.x` is configured to *always* fail with "diagonalization error", regardless of input.
2.  **Configuration**: Max retries set to 3.
3.  **Execution**: Run job.
4.  **Monitoring**:
    -   Attempt 1 -> Fail -> Fix A.
    -   Attempt 2 -> Fail -> Fix B.
    -   Attempt 3 -> Fail -> Fix C.
    -   Attempt 4 -> Fail -> No more fixes.
    -   *Log*: "Job failed permanently after 3 retries."
5.  **Verification**:
    -   Database status is `FAILED`.
    -   Error reason in DB: "Max retries exceeded (Last error: diagonalization error)".

**Success Criteria:**
-   The loop terminates exactly after max retries.
-   The status is explicitly `FAILED`, not stuck in `Running`.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Auto-Recovery of DFT Jobs
  As an Autonomous Agent
  I want to detect and fix DFT errors
  So that the pipeline continues without human help

  Background:
    Given a structure with ID 500 is pending
    And the default mixing beta is 0.7

  Scenario: Recover from Convergence Failure
    Given the first execution of "pw.x" returns exit code 1
    And the output contains "convergence not achieved"
    When the recovery handler processes the error
    Then it should suggest setting "mixing_beta" to 0.3
    And the runner should re-submit the job with the new parameter
    And the new input file should contain "mixing_beta = 0.3"

  Scenario: Recover from Cholesky Error
    Given the execution fails with "Error in routine cdiaghg"
    When the recovery handler processes the error
    Then it should suggest setting "diagonalization" to "cg" (Conjugate Gradient)
    And the runner should re-submit the job

  Scenario: Timeout Handling
    Given the job configuration has a timeout of 1 hour
    When the execution exceeds 1 hour
    Then the process should be killed
    And the error should be recorded as "OUT_OF_TIME"
    And the recovery handler should suggest increasing temperature (if applicable) or give up
```

# Cycle 05 Specification: Automated DFT Factory - Advanced Recovery (Module C Part 2)

## 1. Summary

In Cycle 04, we built a runner that works when everything goes perfectly. But in computational materials science, things rarely go perfectly. "Convergence not achieved," "Cholesky factorization error," and "Out of walltime" are daily occurrences. In Cycle 05, we implement the **Auto-Recovery Logic** that makes the system resilient.

This module acts as a "Robot Operator". When a QE job fails, instead of giving up, the `RecoveryHandler` analyzes the error message. It then consults a "Strategy Ladder"—a predefined sequence of parameter adjustments (e.g., reduce mixing beta, increase temperature, switch diagonalization algorithm) designed to salvage the calculation. It modifies the input and retries the job. Only after exhausting all strategies does it mark the job as `FAILED`.

Additionally, we implement "Zombie Process Management" to kill jobs that hang indefinitely, ensuring that a single bad job doesn't clog the entire compute cluster.

## 2. System Architecture

Files marked in **bold** are new or modified in this cycle.

### 2.1. File Structure

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── dft/
│       │   ├── runner.py               # Modified to use RecoveryHandler
│       │   └── **recovery.py**         # Error Analysis & Strategy Logic
│       └── utils/
│           └── **process.py**          # Process Management (Kill/Timeout)
└── tests/
    └── dft/
        └── **test_recovery.py**
```

### 2.2. Code Blueprints

#### `src/mlip_autopipec/dft/recovery.py`
The brain of the error handling.

```python
from enum import Enum
from typing import Optional, Dict, Any

class DFTError(Enum):
    CONVERGENCE_FAIL = "convergence_not_achieved"
    DIAGONALIZATION_FAIL = "diagonalization_error"
    OUT_OF_TIME = "maximum_cpu_time_exceeded"
    UNKNOWN = "unknown_error"

class RecoveryHandler:
    def analyze_error(self, stdout: str, stderr: str) -> DFTError:
        """Parses logs to identify the root cause."""
        if "convergence not achieved" in stdout:
            return DFTError.CONVERGENCE_FAIL
        # ... other checks
        return DFTError.UNKNOWN

    def suggest_fix(self, error: DFTError, current_params: Dict[str, Any], attempt: int) -> Optional[Dict[str, Any]]:
        """
        Returns updated parameters based on the strategy ladder.
        Returns None if no more fixes are available (give up).
        """
        if error == DFTError.CONVERGENCE_FAIL:
            if attempt == 1:
                return {"mixing_beta": 0.3} # Try reduced mixing
            elif attempt == 2:
                return {"mixing_beta": 0.1, "mixing_mode": "local-tf"}
            elif attempt == 3:
                return {"degauss": current_params.get("degauss", 0.02) + 0.02} # Heat it up

        return None
```

#### `src/mlip_autopipec/dft/runner.py` (Update)
Updating the runner to loop through attempts.

```python
    def run_job(self, row_id: int):
        # ... setup ...
        max_retries = 5
        params = self.default_params.copy()

        for attempt in range(max_retries):
            # Write input with `params`
            # Execute
            success = self.parser.is_success(stdout)

            if success:
                # Save and break
                return

            error = self.recovery_handler.analyze_error(stdout, stderr)
            new_params = self.recovery_handler.suggest_fix(error, params, attempt + 1)

            if new_params:
                params.update(new_params)
                logging.info(f"Job {row_id} failed with {error}. Retrying with {new_params}")
            else:
                logging.error(f"Job {row_id} failed. No more recovery strategies.")
                break

        # Mark as FAILED if loop finishes
```

## 3. Design Architecture

### 3.1. Domain Concepts

1.  **The Strategy Ladder**: Recovery is not random; it's hierarchical. We try cheap/safe fixes first (mixing beta) before trying drastic ones (increasing temperature, which physically alters the system).
2.  **The "Zombie"**: A process that is running but producing no output or has exceeded its walltime. We use python's `subprocess.run(timeout=...)` to handle this at the OS level.

### 3.2. Consumers and Producers

-   **Consumer**: `RecoveryHandler` consumes text logs (stdout/stderr).
-   **Producer**: `RecoveryHandler` produces a `dict` of parameter overrides.

## 4. Implementation Approach

### Step 1: Error Classification
-   **Task**: Create a library of common QE error strings.
-   **Resource**: Consult QE forums and experience. Common ones:
    -   `c_bands:  X eigenvalues not converged`
    -   `error in routine cdiaghg`
    -   `charge density is not correct`

### Step 2: Strategy Implementation
-   **Task**: Implement the ladder logic.
-   **Logic**:
    -   Level 1: `mixing_beta` 0.7 -> 0.3.
    -   Level 2: `mixing_mode` plain -> local-tf.
    -   Level 3: `diagonalization` david -> cg.
    -   Level 4: `smearing` +0.01 Ry (Use with caution).

### Step 3: Runner Loop Refactoring
-   **Task**: Modify `QERunner` to support a `while` or `for` loop.
-   **Task**: Ensure that the input file is *overwritten* or a new one is created (e.g., `pw.retry1.in`) for debugging history.

## 5. Test Strategy

### 5.1. Unit Testing Approach (Min 300 words)

-   **Log Parsing**:
    -   *Test*: Feed a fake log file containing "Error in routine cdiaghg (1)".
    -   *Assert*: `analyze_error` returns `DIAGONALIZATION_FAIL`.
-   **Strategy Ladder**:
    -   *Test*: Call `suggest_fix` with `CONVERGENCE_FAIL` and `attempt=1`. Assert `mixing_beta` is 0.3.
    -   *Test*: Call `suggest_fix` with `attempt=10`. Assert it returns `None` (Abort).
-   **Parameter Merging**:
    -   *Test*: Ensure that the new parameters returned by `suggest_fix` correctly override the defaults without deleting other necessary params (like `pseudopotential_dir`).

### 5.2. Integration Testing Approach (Min 300 words)

-   **Simulated Failure Loop**:
    -   We need a "mock pw.x" that is programmable.
    -   *Setup*: Create a mock script that fails 2 times with "convergence not achieved" and succeeds on the 3rd time *only if* `mixing_beta` is found in the input file as 0.3.
    -   *Execution*: Run `QERunner`.
    -   *Expectation*:
        -   Attempt 1: Fails (Standard params).
        -   Runner calls Recovery. Gets `mixing_beta=0.3`.
        -   Attempt 2: Fails (Mock logic says "fail twice").
        -   Runner calls Recovery. Gets `mixing_mode=local-tf`.
        -   Attempt 3: Succeeds (Mock logic sees updated input).
    -   *Result*: Job marked as `COMPLETED`. Status history shows retries.
-   **Timeout Test**:
    -   *Setup*: Mock script sleeps for 10 seconds. Timeout set to 1 second.
    -   *Execution*: Run job.
    -   *Expectation*: `subprocess.TimeoutExpired` is caught. Recovery handler might suggest increasing time or killing. Job marked `FAILED` (or retried).

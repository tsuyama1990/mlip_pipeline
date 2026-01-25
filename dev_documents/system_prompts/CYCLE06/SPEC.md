# Cycle 06 Specification: Validation Suite

## 1. Summary

Cycle 06 implements the "Validator" or "Gatekeeper" module. In automated materials discovery, it is easy to generate potentials that have low RMSE (numerical error) but exhibit non-physical behavior, such as imaginary phonon modes (indicating dynamic instability) or negative bulk moduli.

This cycle adds a suite of physical tests that run after every training cycle.
1.  **Phonon Stability**: Calculates phonon dispersion to ensure no imaginary frequencies exist for stable phases.
2.  **Elastic Stability**: Calculates the elastic stiffness tensor ($C_{ij}$) and verifies Born stability criteria.
3.  **Equation of State (EOS)**: Verifies that the energy vs. volume curve behaves correctly (positive convexity).

If a potential fails these tests, it is rejected, preventing the pollution of the production environment with bad models.

## 2. System Architecture

**Files to be created/modified in this cycle are marked in Bold.**

```ascii
src/mlip_autopipec/
├── validation/
│   ├── **__init__.py**
│   ├── **runner.py**           # Orchestrates all tests
│   ├── **phonons.py**          # Phonon calculations (Phonopy/ASE)
│   ├── **elastic.py**          # Elastic constant calculations
│   └── **eos.py**              # EOS fitting and checks
```

## 3. Design Architecture

### `ValidationRunner` (in `validation/runner.py`)
*   **Responsibility**: Run all configured validators and aggregate results.
*   **Logic**:
    *   Accepts a `potential_path` and `ValidationConfig`.
    *   Returns a `ValidationReport` object (PASS/FAIL/WARN).

### `PhononValidator` (in `validation/phonons.py`)
*   **Responsibility**: Check dynamic stability.
*   **Logic**:
    *   Uses `ase.phonons` or interfaces with `phonopy`.
    *   Calculates the band structure.
    *   **Fail Condition**: Any frequency $\omega < -0.1$ THz (allow small tolerance for gamma point).

### `ElasticValidator` (in `validation/elastic.py`)
*   **Responsibility**: Check mechanical stability.
*   **Logic**:
    *   Applies small strains ($\pm 1\%$) to the unit cell.
    *   Calculates stress tensor.
    *   Fits $C_{ij}$ matrix.
    *   Checks Born criteria (e.g., for cubic: $C_{11}-C_{12} > 0$, $C_{11} + 2C_{12} > 0$, $C_{44} > 0$).

### `EOSValidator` (in `validation/eos.py`)
*   **Responsibility**: Check thermodynamic stability.
*   **Logic**:
    *   Scans volume $\pm 10\%$.
    *   Fits Birch-Murnaghan EOS.
    *   **Fail Condition**: Bulk modulus $B_0 \le 0$ or fitting error too high.

## 4. Implementation Approach

1.  **EOS**: Simplest to implement using `ase.eos`.
2.  **Elastic**: Implement finite-difference method for $C_{ij}$.
3.  **Phonon**: Most complex. Requires `phonopy` as a dependency. If `phonopy` is too heavy, use `ase.phonons` for a lightweight approximation (Gamma point only or simple supercell). *Decision: Use ASE's internal phonon calculator for simplicity in Cycle 06, upgrade to Phonopy in Cycle 07 if needed.*
4.  **Runner**: Integrate into the workflow.

## 5. Test Strategy

### Unit Testing
*   **EOS**: Feed a set of E-V points from a known LJ potential. Verify $B_0$ is positive.
*   **Elastic**: Feed a known cubic crystal (e.g., Al). Verify $C_{11}, C_{12}, C_{44}$ match literature values (approx).
*   **Phonon**: Feed a stable crystal (Si). Verify no imaginary modes. Feed an unstable structure. Verify failure.

### Integration Testing
*   **Gatekeeper**:
    *   Mock a potential that fails Elastic check.
    *   Run `ValidationRunner`.
    *   Assert output status is `FAIL`.

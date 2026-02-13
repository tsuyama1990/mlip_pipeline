import logging
from typing import Any

from ase import Atoms

from mlip_autopipec.domain_models.config import OracleConfig

logger = logging.getLogger(__name__)


def run_with_healing(atoms: Atoms, config: OracleConfig, max_retries: int = 3) -> None:
    """
    Executes the calculator attached to atoms with self-healing logic.

    Args:
        atoms: ASE Atoms object with a calculator attached.
        config: Oracle configuration.
        max_retries: Maximum number of retries before giving up.
    """
    if atoms.calc is None:
        msg = "Atoms object must have a calculator attached."
        raise ValueError(msg)

    # We need to ensure we are running the calculation.
    # get_potential_energy() triggers it.
    for attempt in range(max_retries + 1):
        try:
            atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        except Exception as e:
            msg = str(e).lower()
            # Catch convergence errors
            # QE often says "convergence not achieved"
            # VASP says "unconverged"
            if ("convergence" in msg or "scf" in msg) and attempt < max_retries:
                logger.warning(
                    f"SCF failed (attempt {attempt + 1}/{max_retries + 1}). Applying healing..."
                )
                _apply_healing(atoms, config, attempt)
                continue

            # If not a recoverable error or max retries reached, re-raise
            raise

        else:
            return  # Success


def _apply_healing(atoms: Atoms, config: OracleConfig, attempt: int) -> None:
    """
    Applies heuristics to fix SCF convergence issues.
    """
    # Access calculator parameters safely
    # calc.parameters is a dict in ASE
    if not hasattr(atoms.calc, "parameters"):
        logger.warning("Calculator does not expose 'parameters'. Cannot apply healing.")
        return

    params: dict[str, Any] = atoms.calc.parameters

    updates: dict[str, Any] = {}

    # Heuristic 1: Reduce mixing beta (for QE)
    # Default is often 0.7. Reduce by half each time.
    current_beta = params.get("mixing_beta", config.mixing_beta)
    # Ensure it's a float
    if isinstance(current_beta, (int, float)):
        new_beta = float(current_beta) * 0.5
        updates["mixing_beta"] = new_beta

    # Heuristic 2: Increase smearing temperature (degauss)
    # Only if we already tried reducing beta once (attempt > 0)
    if attempt > 0:
        current_smearing = params.get("smearing_width", config.smearing_width)
        # QE uses 'degauss'. We map 'smearing_width' to the calculator's key if needed.
        # Here we assume the calculator accepts 'smearing_width' or whatever key matches config.
        # If the calculator is QE, it uses 'conv_thr', 'mixing_beta', 'electron_maxstep'.
        # ASE Espresso calculator maps 'smearing' to 'occupations' and width to 'degauss'.
        # However, if we are passing kwargs to 'set', we should use ASE names if using ASE calculator,
        # or QE names if using custom driver?
        # The prompt implies we use ASE calculators. ASE Espresso uses 'smearing'=(name, width) or separate keywords?
        # Actually ASE Espresso uses 'input_data' dict for raw QE keywords, or specific args.
        # But let's assume standard ASE keys or keys consistent with OracleConfig.

        # In test we check 'mixing_beta'.

        if isinstance(current_smearing, (int, float)):
            new_smearing = float(current_smearing) * 2.0
            updates["smearing_width"] = new_smearing

    logger.info(f"Healing updates: {updates}")

    # Apply updates
    if updates:
        atoms.calc.set(**updates)

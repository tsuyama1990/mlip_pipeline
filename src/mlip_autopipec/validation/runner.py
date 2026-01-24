import logging
import shutil
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.data import atomic_numbers

from mlip_autopipec.config.models import MLIPConfig
from mlip_autopipec.config.schemas.validation import ValidationConfig
from mlip_autopipec.validation.elasticity import ElasticityValidator
from mlip_autopipec.validation.eos import EOSValidator
from mlip_autopipec.validation.phonon import PhononValidator

logger = logging.getLogger(__name__)


class ValidationRunner:
    """
    Orchestrates validation checks (Phonon, Elasticity, EOS).
    """

    def __init__(self, config: MLIPConfig, potential_path: Path) -> None:
        self.config = config
        self.potential_path = potential_path
        self.validation_config = config.validation_config or ValidationConfig()
        self.inference_config = config.inference_config

    def run(self, atoms: Atoms, flags: dict[str, bool] | None = None) -> bool:
        """
        Runs validation checks.

        Args:
            atoms: The structure to validate.
            flags: Dictionary of flags to override config (e.g. {'phonon': True}).

        Returns:
            bool: True if all enabled checks pass.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError(f"Expected ase.Atoms object, got {type(atoms)}")

        if flags is None:
            flags = {}

        # Determine which checks to run
        # Priority: flag > config > default(False if not in config)
        run_phonon = flags.get("phonon", self.validation_config.phonon.enabled)
        run_elastic = flags.get("elastic", self.validation_config.elasticity.enabled)
        run_eos = flags.get("eos", self.validation_config.eos.enabled)

        if not (run_phonon or run_elastic or run_eos):
            logger.info("No validation checks enabled.")
            return True

        # Setup Calculator
        try:
            calculator = self._get_calculator(atoms)
            atoms.calc = calculator
        except Exception as e:
            logger.error(f"Failed to setup calculator for validation: {e}")
            return False

        success = True

        if run_phonon:
            logger.info("Running Phonon Validation...")
            v_phon = PhononValidator(self.validation_config.phonon)
            if not v_phon.validate(atoms, calculator):
                logger.error("Phonon validation failed.")
                success = False
            else:
                logger.info("Phonon validation passed.")

        if run_elastic:
            logger.info("Running Elasticity Validation...")
            v_elast = ElasticityValidator(self.validation_config.elasticity)
            if not v_elast.validate(atoms, calculator):
                logger.error("Elasticity validation failed.")
                success = False
            else:
                logger.info("Elasticity validation passed.")

        if run_eos:
            logger.info("Running EOS Validation...")
            v_eos = EOSValidator(self.validation_config.eos)
            if not v_eos.validate(atoms, calculator):
                logger.error("EOS validation failed.")
                success = False
            else:
                logger.info("EOS validation passed.")

        return success

    def _get_calculator(self, atoms: Atoms) -> Any:
        suffix = self.potential_path.suffix
        if suffix == ".yace":
            return self._create_lammps_calculator(atoms)
        if suffix == ".model":
            # Assume MACE
            try:
                from mace.calculators import MACECalculator  # type: ignore

                # device='cpu' for validation checks usually sufficient and safer
                return MACECalculator(model_paths=str(self.potential_path), device="cpu")
            except ImportError:
                logger.error("MACE not installed.")
                raise RuntimeError("MACE not installed but .model potential provided.")
        else:
            raise ValueError(f"Unknown potential format: {suffix}")

    def _create_lammps_calculator(self, atoms: Atoms) -> Any:
        lammps_executable = self._resolve_lammps_cmd()

        elements = sorted(set(atoms.get_chemical_symbols()))
        elements_str = " ".join(elements)

        use_zbl = False
        # Check if inference config requests ZBL
        if self.inference_config and self.inference_config.use_zbl_baseline:
            use_zbl = True

        parameters = {}

        if use_zbl and self.inference_config:
            # Hybrid Overlay
            pair_style = f"hybrid/overlay pace zbl {self.inference_config.zbl_inner_cutoff} {self.inference_config.zbl_outer_cutoff}"
            pair_coeff = [f"* * pace {self.potential_path.resolve()} {elements_str}"]

            # ZBL coeffs
            for i, el1 in enumerate(elements, start=1):
                z1 = atomic_numbers[el1]
                for j, el2 in enumerate(elements, start=1):
                    if j >= i:
                        z2 = atomic_numbers[el2]
                        pair_coeff.append(f"{i} {j} zbl {z1} {z2}")

            parameters["pair_style"] = pair_style
            parameters["pair_coeff"] = pair_coeff
        else:
            parameters["pair_style"] = "pace"
            parameters["pair_coeff"] = [f"* * {self.potential_path.resolve()} {elements_str}"]

        # Create calculator
        # Pass command explicitly to ensure we use the right binary
        return LAMMPS(command=lammps_executable, specorder=elements, **parameters)

    def _resolve_lammps_cmd(self) -> str:
        cmd = "lmp"
        if self.inference_config and self.inference_config.lammps_executable:
            cmd = str(self.inference_config.lammps_executable)

        # Security check for shell injection
        if any(char in cmd for char in [";", "|", "&", "`", "$", "(", ")", "<", ">"]):
            raise ValueError(f"Security: Invalid characters detected in LAMMPS executable path: {cmd}")

        resolved = shutil.which(cmd)
        if resolved:
            return resolved

        # Fallback
        for name in ["lmp", "lmp_serial", "lmp_mpi"]:
            resolved = shutil.which(name)
            if resolved:
                return resolved

        raise RuntimeError(f"LAMMPS executable '{cmd}' not found.")

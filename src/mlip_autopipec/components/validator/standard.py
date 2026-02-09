import logging
import tempfile
from pathlib import Path

from ase.build import bulk
from ase.data import chemical_symbols
from ase.filters import UnitCellFilter
from ase.optimize import BFGS

from mlip_autopipec.components.validator.base import BaseValidator
from mlip_autopipec.components.validator.calculator import LammpsSinglePointCalculator
from mlip_autopipec.components.validator.elastic import ElasticCalc
from mlip_autopipec.components.validator.eos import EOSCalc
from mlip_autopipec.components.validator.phonons import PhononCalc
from mlip_autopipec.domain_models.config import StandardValidatorConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.results import ValidationMetrics
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class StandardValidator(BaseValidator):
    """
    Standard implementation of the Validator component.
    """

    def __init__(self, config: StandardValidatorConfig) -> None:
        super().__init__(config)
        self.config: StandardValidatorConfig = config

        # Instantiate sub-validators with config
        # Default displacement from config if available (future improvement), strictly 0.01 for now as per requirement
        self.phonon_calc = PhononCalc(
            supercell_matrix=self.config.phonon_supercell,
            displacement=self.config.phonon_displacement,
        )
        self.elastic_calc = ElasticCalc(
            strain_magnitude=self.config.elastic_strain_magnitude
        )
        self.eos_calc = EOSCalc(
            strain_range=self.config.eos_strain_range,
            n_points=7
        )

    @property
    def name(self) -> str:
        return self.config.name

    def validate(self, potential: Potential) -> ValidationMetrics:
        """
        Validate the potential.
        """
        # Heuristic structure selection (FCC/BCC/Diamond)
        species = potential.species
        if not species:
            logger.warning("Potential has no species defined. Skipping physical validation.")
            return ValidationMetrics(passed=True, details={"warning": "No species"})

        element = species[0]
        # Validate element symbol
        if element not in chemical_symbols:
             logger.error(f"Invalid element symbol: {element}")
             return ValidationMetrics(passed=False, details={"error": f"Invalid element: {element}"})

        # Structure selection using configuration
        struct_type = self.config.structure_map.get(element, 'fcc')

        try:
            atoms = bulk(element, struct_type, cubic=True)
            structure = Structure.from_ase(atoms)
        except Exception:
            # Fallback
            atoms = bulk(element, 'fcc', cubic=True)
            structure = Structure.from_ase(atoms)

        # Context manager for temp dir ensures cleanup
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)

            # Setup Calculator
            calc = LammpsSinglePointCalculator(
                potential=potential,
                workdir=workdir,
                keep_files=False,
                template_units=self.config.lammps_template_units,
                template_atom_style=self.config.lammps_template_atom_style,
                template_boundary=self.config.lammps_template_boundary,
            )

            # 1. Relax the structure
            atoms_to_relax = structure.to_ase()
            atoms_to_relax.calc = calc

            try:
                ucf = UnitCellFilter(atoms_to_relax) # type: ignore[no-untyped-call]
                opt = BFGS(ucf, logfile=None) # type: ignore[arg-type, no-untyped-call]
                opt.run(fmax=0.01, steps=50) # type: ignore[no-untyped-call]
                relaxed_structure = Structure.from_ase(atoms_to_relax)
            except Exception as e:
                logger.exception("Relaxation failed")
                return ValidationMetrics(passed=False, details={"error": f"Relaxation failed: {e}"})

            # 2. Phonons
            is_phonon_stable, failed_phonon_struct = self.phonon_calc.calculate(
                relaxed_structure, calc, workdir / "phonons"
            )

            # 3. Elastic
            is_elastic_stable, B, G = self.elastic_calc.calculate(
                relaxed_structure, calc, workdir / "elastic"
            )

            # 4. EOS
            eos_rmse = self.eos_calc.calculate(
                relaxed_structure, calc, workdir / "eos"
            )

            # Aggregate results
            failed_structures = []
            if failed_phonon_struct:
                failed_structures.append(failed_phonon_struct)

            passed = is_phonon_stable and is_elastic_stable and (eos_rmse < 0.1)

            return ValidationMetrics(
                passed=passed,
                phonon_stable=is_phonon_stable,
                elastic_stable=is_elastic_stable,
                bulk_modulus=B,
                shear_modulus=G,
                eos_rmse=eos_rmse,
                failed_structures=failed_structures,
                details={
                    "structure_used": f"{element} {struct_type}",
                    "eos_rmse_threshold": 0.1
                }
            )

    def __repr__(self) -> str:
        return f"<StandardValidator(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"StandardValidator({self.name})"

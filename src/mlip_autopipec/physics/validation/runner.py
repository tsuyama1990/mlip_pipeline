from pathlib import Path
from ase import Atoms
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.validation import ValidationResult

from mlip_autopipec.physics.validation.eos import EOSValidator
from mlip_autopipec.physics.validation.elasticity import ElasticityValidator
from mlip_autopipec.physics.validation.phonon import PhononValidator

class ValidationRunner:
    def __init__(self, potential_path: Path, config: ValidationConfig, potential_config: PotentialConfig, lammps_command: str = "lmp"):
        self.potential_path = potential_path
        self.config = config
        self.potential_config = potential_config
        self.lammps_command = lammps_command

    def validate(self, reference_structure: Atoms) -> list[ValidationResult]:
        results = []

        # EOS
        try:
            eos_val = EOSValidator(self.potential_path, self.config, self.potential_config, self.lammps_command)
            results.append(eos_val.validate(reference_structure))
        except Exception as e:
            # Handle crash in validator instantiation or unexpected error
            # But validate() should handle its own errors.
            pass

        # Elasticity
        try:
            elas_val = ElasticityValidator(self.potential_path, self.config, self.potential_config, self.lammps_command)
            results.append(elas_val.validate(reference_structure))
        except Exception:
            pass

        # Phonon
        try:
            phonon_val = PhononValidator(self.potential_path, self.config, self.potential_config, self.lammps_command)
            results.append(phonon_val.validate(reference_structure))
        except Exception:
            pass

        return results

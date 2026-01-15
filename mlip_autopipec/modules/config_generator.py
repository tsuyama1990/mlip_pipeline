# ruff: noqa: D101, D102, D103, D107
"""Module for generating training configuration files."""

from pathlib import Path

import yaml

from mlip_autopipec.config_schemas import SystemConfig


class PacemakerConfigGenerator:
    """Generates configuration files for the Pacemaker training engine."""

    def __init__(self, config: SystemConfig):
        self.config = config

    def generate_config(self, data_file_path: Path, output_dir: Path) -> Path:
        """Generate the Pacemaker YAML config file from the SystemConfig.

        Args:
            data_file_path: The path to the training data file.
            output_dir: The directory where the config file will be saved.

        Returns:
            The path to the generated YAML configuration file.

        """
        trainer_params = self.config.trainer
        config_dict = {
            "fit_params": {
                "dataset_filename": str(data_file_path),
                "loss_weights": {
                    "energy": trainer_params.loss_weights.energy,
                    "forces": trainer_params.loss_weights.forces,
                    "stress": trainer_params.loss_weights.stress,
                },
                "ace": {
                    "radial_basis": trainer_params.ace_params.radial_basis,
                    "correlation_order": trainer_params.ace_params.correlation_order,
                    "element_dependent_cutoffs": (
                        trainer_params.ace_params.element_dependent_cutoffs
                    ),
                },
            }
        }

        config_file = output_dir / "pacemaker_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        return config_file

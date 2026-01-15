"""Module for generating training configuration files."""

from pathlib import Path

import yaml

from mlip_autopipec.config_schemas import SystemConfig


class PacemakerConfigGenerator:
    """Generates configuration files for the Pacemaker training engine.

    This class reads the training parameters from a `SystemConfig` object and
    translates them into the specific YAML format required by the `pacemaker_train`
    command-line tool.

    Attributes:
        config: The system configuration object containing the trainer parameters.

    """

    def __init__(self, config: SystemConfig):
        self.config = config

    def generate_config(self, data_file_path: Path, output_dir: Path) -> Path:
        """Generate the Pacemaker YAML config file.

        This method constructs a Python dictionary that mirrors the structure of the
        required YAML file and then writes it to the specified output directory.

        Args:
            data_file_path: The path to the training data file (e.g., 'data.xyz').
            output_dir: The directory where the config file will be saved.

        Returns:
            The path to the newly generated YAML configuration file.

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

"""Module for generating training configuration files."""

from pathlib import Path

import yaml

from mlip_autopipec.config.models import SystemConfig


class PacemakerConfigGenerator:
    """
    Generates configuration files for the Pacemaker training engine.

    This class is responsible for translating the high-level training parameters
    defined in the `SystemConfig` into the specific YAML format required by the
    `pacemaker_train` command-line tool. It ensures that the generated
    configuration is valid and ready for use in a training job.
    """

    def __init__(self, config: SystemConfig):
        """
        Initializes the PacemakerConfigGenerator.

        Args:
            config: A fully validated `SystemConfig` object containing all the
                    necessary parameters for the training job.
        """
        self.config = config

    def generate_config(self, data_file_path: Path, output_dir: Path) -> Path:
        """
        Generates the Pacemaker YAML config file.

        This method constructs a Python dictionary that mirrors the structure of the
        required YAML file, populating it with values from the `SystemConfig`.
        It then writes this dictionary to a YAML file in the specified output
        directory.

        Args:
            data_file_path: The absolute path to the training data file
                            (e.g., '/path/to/training_data.xyz'). This path
                            is written directly into the configuration file.
            output_dir: The directory where the generated 'pacemaker_config.yaml'
                        will be saved.

        Returns:
            The absolute path to the newly generated YAML configuration file.
        """
        trainer_params = self.config.training_config
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
                    "element_dependent_cutoffs": trainer_params.ace_params.element_dependent_cutoffs
                },
            }
        }

        from mlip_autopipec.config.models import PacemakerConfig
        PacemakerConfig.model_validate(config_dict)

        config_file = output_dir / "pacemaker_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        return config_file

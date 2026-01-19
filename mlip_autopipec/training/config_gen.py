from pathlib import Path

import jinja2

from mlip_autopipec.config.schemas.training import TrainConfig


class TrainConfigGenerator:
    """Generates Pacemaker configuration files."""

    def __init__(self, template_path: Path) -> None:
        self.template_path = template_path
        if not self.template_path.exists():
            msg = f"Template not found at {self.template_path}"
            raise FileNotFoundError(msg)

    def generate(
        self, config: TrainConfig, data_path: Path, output_path: Path, elements: list[str]
    ) -> Path:
        """
        Generates input.yaml for Pacemaker.

        Args:
            config: The training configuration.
            data_path: Path to the training dataset file (.pckl.gzip).
            output_path: Path where the output potential will be saved (used to naming).
            elements: List of chemical symbols in the system.

        Returns:
            Path to the generated input.yaml file.
        """

        with open(self.template_path) as f:
            template = jinja2.Template(f.read())

        rendered = template.render(
            config=config,
            data_path=str(data_path.absolute()),
            output_path=output_path,
            elements=elements,
        )

        output_yaml = output_path.parent / "input.yaml"
        with open(output_yaml, "w") as f:
            f.write(rendered)

        return output_yaml

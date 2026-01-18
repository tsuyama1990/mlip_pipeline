from pathlib import Path

import pytest
import yaml

from mlip_autopipec.config.schemas.training import TrainConfig
from mlip_autopipec.training.config_gen import TrainConfigGenerator


def test_config_generation(tmp_path):
    # create a dummy template
    template_path = tmp_path / "template.yaml.j2"
    with open(template_path, "w") as f:
        f.write("cutoff: {{ config.cutoff }}\ndata_path: {{ data_path }}\n")

    gen = TrainConfigGenerator(template_path)
    config = TrainConfig(cutoff=4.5)

    data_path = tmp_path / "data.pckl.gzip"
    output_path = tmp_path / "output.yace"

    input_yaml = gen.generate(config, data_path, output_path, elements=["Al", "Cu"])

    assert input_yaml.exists()

    with open(input_yaml) as f:
        content = yaml.safe_load(f)

    assert content["cutoff"] == 4.5
    assert content["data_path"] == str(data_path.absolute())

def test_template_not_found():
    with pytest.raises(FileNotFoundError):
        TrainConfigGenerator(Path("non_existent_template.j2"))

from pathlib import Path

from mlip_autopipec.config.schemas.inference import InferenceConfig, InferenceResult


def test_inference_config_valid():
    config = InferenceConfig(
        temperature=1000.0,
        pressure=0.0,
        timestep=0.002,
        steps=5000,
        ensemble="nvt",
        uq_threshold=4.0,
        sampling_interval=50,
        potential_path=Path("/tmp/model.yace"),
        lammps_executable="lmp_serial",
    )
    # Check flat properties
    assert config.temperature == 1000.0
    assert config.pressure == 0.0
    assert config.timestep == 0.002
    assert config.steps == 5000
    assert config.ensemble == "nvt"
    assert config.uq_threshold == 4.0
    assert config.sampling_interval == 50
    assert config.potential_path == Path("/tmp/model.yace")
    assert config.lammps_executable == "lmp_serial"


def test_inference_config_invalid_executable():
    # If using Path, it must be executable (if it exists)
    pass


def test_inference_result_schema():
    res = InferenceResult(
        succeeded=True,
        final_structure=Path("/tmp/final.data"),
        uncertain_structures=[Path("/tmp/u1.data"), Path("/tmp/u2.data")],
        max_gamma_observed=4.5,
    )
    assert res.succeeded
    assert len(res.uncertain_structures) == 2
    assert res.max_gamma_observed == 4.5


def test_inference_result_defaults():
    res = InferenceResult(succeeded=False, max_gamma_observed=15.0)
    assert res.final_structure is None
    assert res.uncertain_structures == []

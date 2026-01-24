from ase.build import bulk

from mlip_autopipec.config.schemas.generator import SQSConfig
from mlip_autopipec.generator.sqs import SQSStrategy


def test_sqs_generation_random_shuffle() -> None:
    # Test fallback logic (random shuffle) since icet might not be present or we force fallback
    config = SQSConfig(enabled=True, supercell_size=[2, 2, 2])
    strategy = SQSStrategy(config, seed=42)

    prim = bulk("Au")
    composition = {"Au": 0.5, "Cu": 0.5}

    sqs = strategy.generate(prim, composition)

    # Check size
    assert len(sqs) == len(prim) * 8  # 2x2x2 = 8

    # Check composition
    symbols = sqs.get_chemical_symbols()
    n_Au = symbols.count("Au")
    n_Cu = symbols.count("Cu")

    assert n_Au == 4
    assert n_Cu == 4
    assert sqs.info["config_type"] == "sqs"


def test_sqs_generation_composition_rounding() -> None:
    config = SQSConfig(enabled=True, supercell_size=[2, 2, 2])
    strategy = SQSStrategy(config)

    prim = bulk("Au")
    # 8 atoms. 0.3 * 8 = 2.4 -> 2.
    # 0.7 * 8 = 5.6 -> 6.
    # Total 8.
    composition = {"Au": 0.3, "Cu": 0.7}

    sqs = strategy.generate(prim, composition)
    symbols = sqs.get_chemical_symbols()
    assert symbols.count("Au") == 2
    assert symbols.count("Cu") == 6

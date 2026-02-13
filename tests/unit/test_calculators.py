from pathlib import Path
from unittest.mock import MagicMock, patch

from ase.calculators.emt import EMT

from mlip_autopipec.dynamics.calculators import MLIPCalculatorFactory


def test_calculator_factory_emt_fallback() -> None:
    factory = MLIPCalculatorFactory()
    calc = factory.create(Path("unknown.pot"))
    assert isinstance(calc, EMT)

def test_calculator_factory_pace() -> None:
    factory = MLIPCalculatorFactory()
    path = Path("test.yace")

    # If we want to test success, we must mock the import.
    with patch.dict("sys.modules", {"pyace": MagicMock()}), \
         patch("mlip_autopipec.dynamics.calculators.MLIPCalculatorFactory._create_pace") as mock_create:
             mock_create.return_value = MagicMock()
             _ = factory.create(path)
             mock_create.assert_called_with(path)

def test_calculator_factory_m3gnet(tmp_path: Path) -> None:
    factory = MLIPCalculatorFactory()
    # Mock m3gnet import
    with patch.dict("sys.modules", {"m3gnet": MagicMock(), "m3gnet.models": MagicMock(), "m3gnet.calculators": MagicMock()}), \
         patch("mlip_autopipec.dynamics.calculators.MLIPCalculatorFactory._create_m3gnet") as mock_create:
             mock_create.return_value = MagicMock()
             _ = factory.create(Path("model.m3gnet"))
             mock_create.assert_called()

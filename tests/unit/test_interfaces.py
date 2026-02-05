from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator

def test_mocks_implement_interfaces() -> None:
    assert isinstance(MockExplorer(), Explorer)
    assert isinstance(MockOracle(), Oracle)
    assert isinstance(MockTrainer(), Trainer)
    assert isinstance(MockValidator(), Validator)

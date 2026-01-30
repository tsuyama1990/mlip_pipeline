import pytest
from mlip_autopipec.physics.dft.parser import DFTParser
from mlip_autopipec.domain_models.calculation import DFTError, SCFError, WalltimeError, MemoryError, DFTConfig
from mlip_autopipec.physics.dft.recovery import RecoveryHandler

def test_parser_missing_energy():
    parser = DFTParser()
    output = "Some random output without energy."
    with pytest.raises(DFTError, match="Job finished but no energy found"):
        parser.parse(output)

def test_parser_missing_forces():
    parser = DFTParser()
    output = """
!    total energy              =     -15.78901234 Ry
    """
    with pytest.raises(DFTError, match="Forces not found"):
        parser.parse(output)

def test_parser_walltime_error():
    parser = DFTParser()
    output = "maximum CPU time exceeded"
    with pytest.raises(WalltimeError):
        parser.parse(output)

def test_parser_memory_error():
    parser = DFTParser()
    output = "Error in routine allocation: failed"
    with pytest.raises(MemoryError):
        parser.parse(output)

def test_parser_unknown_routine_error():
    parser = DFTParser()
    output = "Error in routine unknown (1): some unknown error"
    with pytest.raises(DFTError, match="QE Error: some unknown error"):
        parser.parse(output)

def test_parser_time_parsing():
    parser = DFTParser()
    assert parser._parse_time("1h20m30s") == 3600 + 1200 + 30
    assert parser._parse_time("1m30s") == 90
    assert parser._parse_time("30s") == 30
    assert parser._parse_time("1.5") == 1.5

def test_recovery_attempts_3_4_5():
    handler = RecoveryHandler()
    config = DFTConfig(command="pw.x", pseudopotentials={}, ecutwfc=30.0)
    error = SCFError("fail")

    # Attempt 3: Increase smearing
    new_config = handler.apply_fix(config, error, attempt=3)
    # Default degauss 0.02 -> 0.03
    assert new_config.degauss > 0.02

    # Attempt 4: Change diagonalization
    new_config = handler.apply_fix(config, error, attempt=4)
    assert new_config.diagonalization == "cg"

    # Attempt 5: Drastic
    new_config = handler.apply_fix(config, error, attempt=5)
    assert new_config.mixing_beta == 0.1
    assert new_config.degauss == 0.05

def test_recovery_walltime():
    handler = RecoveryHandler()
    config = DFTConfig(command="pw.x", pseudopotentials={}, ecutwfc=30.0, timeout=100)
    error = WalltimeError("timeout")

    new_config = handler.apply_fix(config, error, attempt=1)
    assert new_config.timeout == 150

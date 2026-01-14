from mlip_autopipec.modules.qe_output_parser import QEOutputParser


def test_qe_output_parser() -> None:
    """Test the QEOutputParser."""
    fake_qe_output = """
    !    total energy              =     -123.456 Ry
    Forces acting on atoms:
        atom    1 type  1   force =     0.000000000   0.000000000   0.000000000
    total stress  (Ry/bohr**3)            (kbar)       (GPa)
        -0.00000000   -0.00000000    0.00000000    -0.00      -0.00       0.00
        -0.00000000   -0.00000000    0.00000000    -0.00      -0.00       0.00
         0.00000000    0.00000000    0.00000000     0.00       0.00       0.00
    """
    parser = QEOutputParser(fake_qe_output)
    output = parser.parse()
    assert output.total_energy == -123.456
    assert len(output.forces) == 1
    assert len(output.stress) == 3

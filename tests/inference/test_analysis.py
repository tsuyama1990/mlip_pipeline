from mlip_autopipec.inference.analysis import AnalysisUtils


def test_parse_log_properties(tmp_path):
    log_content = """
Step Temp Press PotEng
0 300.0 1.0 -100.0
100 305.0 1.1 -100.1
200 295.0 0.9 -99.9
Loop time of 10.0 on 1 procs
"""
    log_file = tmp_path / "log.lammps"
    log_file.write_text(log_content)

    analysis = AnalysisUtils(log_file)
    props = analysis.get_properties()

    assert "temperature" in props
    assert "pressure" in props
    assert "potential_energy" in props
    assert abs(props["temperature"] - 300.0) < 10.0

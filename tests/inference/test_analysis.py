from pathlib import Path

import pytest

from mlip_autopipec.inference.analysis import AnalysisUtils


def test_parse_log_thermo(tmp_path: Path) -> None:
    log_file = tmp_path / "log.lammps"
    log_content = """
Step Temp Press
0 300.0 0.0
100 305.0 1.0
200 295.0 -1.0
Loop time of 1.0
"""
    log_file.write_text(log_content)

    analysis = AnalysisUtils(log_file)
    stats = analysis.get_thermo_stats()

    assert stats["temperature_mean"] == 300.0
    assert stats["pressure_mean"] == 0.0
    assert stats["steps"] == 3


def test_analysis_utils_missing_file(tmp_path: Path) -> None:
    log_file = tmp_path / "missing.log"
    analysis = AnalysisUtils(log_file)
    with pytest.raises(FileNotFoundError):
        analysis.get_thermo_stats()

from mlip_autopipec.utils.plotting import create_band_structure_plot, create_line_plot


def test_create_line_plot() -> None:
    html = create_line_plot(
        x_data=[1.0, 2.0, 3.0],
        y_data=[1.0, 4.0, 9.0],
        title="Test Plot",
        x_label="X",
        y_label="Y"
    )
    assert isinstance(html, str)
    assert "Test Plot" in html
    assert "plotly" in html.lower() or "div" in html.lower()


def test_create_band_structure_plot() -> None:
    bands = [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]
    q_points = [0.0, 0.5, 1.0]
    html = create_band_structure_plot(
        frequencies=bands,
        q_points=q_points
    )
    assert isinstance(html, str)
    assert "Phonon Band Structure" in html

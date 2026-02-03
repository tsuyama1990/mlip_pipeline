
import plotly.graph_objects as go


def create_line_plot(
    x_data: list[float],
    y_data: list[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: str | None = None,
) -> str:
    """
    Creates a simple line plot using Plotly.
    Returns the HTML div string.
    If output_path is provided, saves to HTML file.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode="lines"))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )

    if output_path:
        fig.write_html(output_path)

    return fig.to_html(full_html=False, include_plotlyjs="cdn")  # type: ignore[no-any-return]


def create_band_structure_plot(
    frequencies: list[list[float]],
    q_points: list[float],
    title: str = "Phonon Band Structure",
) -> str:
    """
    Creates a band structure plot.
    frequencies: List of bands, where each band is a list of frequencies.
    """
    fig = go.Figure()
    for i, band in enumerate(frequencies):
        fig.add_trace(go.Scatter(x=q_points, y=band, mode="lines", name=f"Band {i}"))

    fig.update_layout(
        title=title,
        xaxis_title="Wave Vector",
        yaxis_title="Frequency (THz)",
        showlegend=False,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")  # type: ignore[no-any-return]

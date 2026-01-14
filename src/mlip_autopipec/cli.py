import typer

from .app import run_pipeline

app = typer.Typer()


@app.command()
def run(config_path: str) -> None:
    """
    Runs the MLIP-AutoPipe pipeline.
    """
    run_pipeline(config_path)


if __name__ == "__main__":
    app()

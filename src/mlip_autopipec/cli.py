from pathlib import Path

import typer

from .app import run_pipeline, train_pipeline

app = typer.Typer()


@app.command()
def run(config_path: str) -> None:
    """
    Runs the MLIP-AutoPipe pipeline.
    """
    run_pipeline(config_path)


@app.command()
def train(config_path: str, database_path: str, output_dir: Path) -> None:
    """
    Trains a new MLIP using a generated dataset.
    """
    train_pipeline(config_path, database_path, output_dir)


if __name__ == "__main__":
    app()

from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from mlip_autopipec.core.config import load_config

app = typer.Typer(no_args_is_help=True)
console = Console()

@app.command()
def check_config(config_path: str) -> None:
    """
    Validates the configuration file.
    """
    path = Path(config_path)
    try:
        if not path.exists():
            console.print(f"[red]Error: Config file {path} not found.[/red]")
            raise typer.Exit(code=1)

        load_config(path)
        console.print("[green]OK[/green]")
    except ValidationError as e:
        console.print(f"[red]Validation Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

@app.command()
def version() -> None:
    """Show version."""
    console.print("0.1.0")

if __name__ == "__main__":
    app()

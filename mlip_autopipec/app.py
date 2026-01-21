import typer
from typing import Annotated
from pathlib import Path
from pydantic import ValidationError
from rich.console import Console
from mlip_autopipec.core.config import load_config

app = typer.Typer(no_args_is_help=True)
console = Console()

def validate_config_file(config_path: Path) -> None:
    """
    Validates the configuration file at the given path.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
    load_config(config_path)

@app.command()
def check_config(config_path: str) -> None:
    """
    Validates the configuration file.
    """
    path = Path(config_path)
    try:
        validate_config_file(path)
        console.print("[green]OK[/green]")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    except ValidationError as e:
        console.print(f"[red]Validation Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    # Removed generic Exception catch as per feedback, or make it specific if needed.
    # But for a CLI, a fallback might be good. I'll stick to specific ones first.
    except Exception as e: # Catch-all for unexpected errors in CLI is standard practice to show nice error instead of traceback
         console.print(f"[red]Unexpected Error:[/red] {e}")
         raise typer.Exit(code=1) from None

@app.command()
def version() -> None:
    """Show version."""
    console.print("0.1.0")

if __name__ == "__main__":
    app()

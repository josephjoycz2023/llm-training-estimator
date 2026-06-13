import yaml
import typer
from rich import print

app = typer.Typer()


@app.command()
def estimate(config: str):
    """Estimate training memory/time/cost from a YAML config."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("[bold green]Loaded config:[/bold green]")
    print(cfg)

    print("\n[bold yellow]Next step:[/bold yellow]")
    print("1. Send model/training/parallelism/hardware config to gpu-mem-calculator.")
    print("2. Send model/gpu/dtype/parallelism/token config to llm-analysis.")
    print("3. Merge memory feasibility, time estimate and cost estimate.")


if __name__ == "__main__":
    app()

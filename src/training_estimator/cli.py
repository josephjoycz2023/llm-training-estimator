from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from training_estimator.config_loader import load_config, load_prices
from training_estimator.coverage import build_coverage_report
from training_estimator.pipeline import run_pipeline
from training_estimator.report import ReportFormat, render_result, write_output

app = typer.Typer(help="Estimate LLM training memory, time, GPU-hours and rental cost.")


@app.command()
def validate(config: Path) -> None:
    """Validate a unified YAML config and print backend coverage."""
    cfg = load_config(config)
    coverage = build_coverage_report(cfg)

    print("[bold green]Config valid.[/bold green]")
    print("")
    print("[bold]Model:[/bold]")
    print(f"  architecture: {cfg.model.architecture}")
    print(f"  total_params_b: {cfg.model.total_params_b}")
    print(f"  active_params_b: {cfg.model.active_params_b}")
    print("")
    print("[bold]Training:[/bold]")
    print(f"  mode: {cfg.training.mode}")
    print(f"  total_tokens_b: {cfg.training.total_tokens_b}")
    print("")
    print("[bold]Backends:[/bold]")
    print(
        "  gpu-mem-calculator: supported"
        if cfg.training.mode == "full"
        else "  gpu-mem-calculator: schema-only for PEFT"
    )
    print(
        "  llm-analysis: supported"
        if cfg.training.mode == "full"
        else "  llm-analysis: schema-only for PEFT"
    )
    if coverage.warnings:
        print("")
        print("[bold yellow]Warnings:[/bold yellow]")
        for warning in coverage.warnings:
            print(f"  - {warning}")


@app.command()
def estimate(
    config: Path,
    prices: Path = typer.Option(
        Path("configs/hardware_prices.yaml"),
        "--prices",
        "-p",
        help="YAML file with monthly rental price profiles.",
    ),
    output_format: ReportFormat = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, or markdown.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write rendered output to this path.",
    ),
) -> None:
    """Run the full pipeline for one YAML config."""
    cfg = load_config(config)
    price_book = load_prices(prices)
    result = run_pipeline(cfg, price_book)
    rendered = render_result(result, output_format)

    if output is not None:
        write_output(output, rendered)
        print(f"[bold green]Wrote output:[/bold green] {output}")
        return

    typer.echo(rendered)


if __name__ == "__main__":
    app()

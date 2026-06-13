from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from training_estimator.schema import FinalResult


ReportFormat = Literal["table", "json", "markdown"]


def render_json(result: FinalResult) -> str:
    return result.model_dump_json(indent=2)


def render_table(result: FinalResult) -> str:
    memory = result.memory
    time = result.time
    cost = result.cost
    lines = [
        f"Run: {result.run_name}",
        "",
        f"Runnable: {str(result.runnable).lower()}",
        f"Risk: {result.risk_level}",
        "",
        "Memory:",
        f"  Backend: {memory.backend}",
        f"  Status: {memory.status}",
        f"  Fits: {memory.fits}",
    ]
    if memory.memory_per_gpu_gb is not None and memory.gpu_memory_limit_gb is not None:
        lines.append(
            f"  Memory per GPU: {memory.memory_per_gpu_gb:.2f} / "
            f"{memory.gpu_memory_limit_gb:.2f} GB"
        )
    if memory.memory_utilization is not None:
        lines.append(f"  Utilization: {memory.memory_utilization * 100:.1f}%")

    if time is not None:
        lines.extend(
            [
                "",
                "Time:",
                f"  Backend: {time.backend}",
                f"  Status: {time.status}",
                f"  Estimate type: {time.estimate_type}",
            ]
        )
        if time.total_time_hours is not None:
            lines.append(f"  Total time: {time.total_time_hours:.2f} h")
        if time.gpu_hours is not None:
            lines.append(f"  GPU-hours: {time.gpu_hours:.2f}")

    if cost is not None:
        lines.extend(
            [
                "",
                "Cost:",
                f"  Price profile: {cost.price_profile}",
                f"  Monthly rental: {cost.monthly_rental_cost:.2f} {cost.currency}",
                f"  Estimated cost: {cost.estimated_cost:.2f} {cost.currency}",
            ]
        )

    lines.extend(["", "Coverage:"])
    for item in result.coverage.supported:
        lines.append(f"  supported: {item}")
    for item in result.coverage.approximated:
        lines.append(f"  approximated: {item}")
    for item in result.coverage.unsupported:
        lines.append(f"  unsupported: {item}")

    if result.warnings:
        lines.extend(["", "Warnings:"])
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


def render_markdown(result: FinalResult) -> str:
    data = json.loads(result.model_dump_json())
    lines = [
        f"# {result.run_name}",
        "",
        f"- Runnable: `{str(result.runnable).lower()}`",
        f"- Risk: `{result.risk_level}`",
        "",
        "## Memory",
        "",
        f"- Backend: `{result.memory.backend}`",
        f"- Status: `{result.memory.status}`",
        f"- Fits: `{result.memory.fits}`",
    ]
    if result.memory.memory_per_gpu_gb is not None:
        lines.append(f"- Memory per GPU: `{result.memory.memory_per_gpu_gb:.2f} GB`")
    if result.memory.memory_utilization is not None:
        lines.append(f"- Utilization: `{result.memory.memory_utilization * 100:.1f}%`")

    if result.time is not None:
        lines.extend(
            [
                "",
                "## Time",
                "",
                f"- Backend: `{result.time.backend}`",
                f"- Status: `{result.time.status}`",
                f"- Estimate type: `{result.time.estimate_type}`",
            ]
        )
        if result.time.total_time_hours is not None:
            lines.append(f"- Total time: `{result.time.total_time_hours:.2f} h`")
        if result.time.gpu_hours is not None:
            lines.append(f"- GPU-hours: `{result.time.gpu_hours:.2f}`")

    if result.cost is not None:
        lines.extend(
            [
                "",
                "## Cost",
                "",
                f"- Price profile: `{result.cost.price_profile}`",
                f"- Monthly rental: `{result.cost.monthly_rental_cost:.2f} {result.cost.currency}`",
                f"- Estimated cost: `{result.cost.estimated_cost:.2f} {result.cost.currency}`",
            ]
        )

    lines.extend(["", "## Coverage", "", "```json", json.dumps(data["coverage"], indent=2), "```"])

    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)

    return "\n".join(lines)


def render_result(result: FinalResult, output_format: ReportFormat) -> str:
    if output_format == "json":
        return render_json(result)
    if output_format == "markdown":
        return render_markdown(result)
    return render_table(result)


def write_output(path: str | Path, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

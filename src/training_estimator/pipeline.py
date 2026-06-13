from __future__ import annotations

from training_estimator.adapters.gpu_mem_adapter import GPUMemCalculatorAdapter
from training_estimator.adapters.llm_analysis_adapter import LLMAnalysisAdapter
from training_estimator.config_loader import PriceBook
from training_estimator.cost import calculate_cost
from training_estimator.coverage import build_coverage_report
from training_estimator.schema import EstimatorConfig, FinalResult, RiskLevel, TimeBackendResult


def risk_from_memory(memory_utilization: float | None, fits: bool | None) -> RiskLevel:
    if fits is False:
        return "high"
    if memory_utilization is None:
        return "unknown"
    if memory_utilization < 0.80:
        return "low"
    if memory_utilization < 0.92:
        return "medium"
    return "high"


def run_pipeline(
    config: EstimatorConfig,
    prices: PriceBook,
    memory_adapter: GPUMemCalculatorAdapter | None = None,
    time_adapter: LLMAnalysisAdapter | None = None,
) -> FinalResult:
    memory_adapter = memory_adapter or GPUMemCalculatorAdapter()
    time_adapter = time_adapter or LLMAnalysisAdapter()

    coverage = build_coverage_report(config)
    memory = memory_adapter.estimate_memory(config)
    warnings = [*coverage.warnings, *memory.warnings]
    risk = risk_from_memory(memory.memory_utilization, memory.fits)

    if memory.status != "ok":
        return FinalResult(
            run_name=config.run.name,
            runnable=False,
            risk_level=risk,
            memory=memory,
            time=None,
            cost=None,
            coverage=coverage,
            warnings=warnings,
        )

    if memory.fits is False:
        time = TimeBackendResult(
            backend="llm-analysis",
            status="not_actionable",
            parsed=False,
            warnings=["memory estimate does not fit; time and cost are not actionable"],
        )
        warnings.extend(time.warnings)
        return FinalResult(
            run_name=config.run.name,
            runnable=False,
            risk_level=risk,
            memory=memory,
            time=time,
            cost=None,
            coverage=coverage,
            warnings=warnings,
        )

    time = time_adapter.estimate_time(config)
    warnings.extend(time.warnings)
    if time.status != "ok":
        return FinalResult(
            run_name=config.run.name,
            runnable=False,
            risk_level=risk,
            memory=memory,
            time=time,
            cost=None,
            coverage=coverage,
            warnings=warnings,
        )

    cost = calculate_cost(config, time, prices)
    if cost is not None:
        warnings.extend(cost.warnings)

    return FinalResult(
        run_name=config.run.name,
        runnable=True,
        risk_level=risk,
        memory=memory,
        time=time,
        cost=cost,
        coverage=coverage,
        warnings=warnings,
    )

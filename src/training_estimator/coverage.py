from __future__ import annotations

from training_estimator.schema import CoverageResult, EstimatorConfig


def build_coverage_report(config: EstimatorConfig) -> CoverageResult:
    supported = [
        "model architecture schema",
        "hardware.gpu_count and hardware.gpu_memory_gb",
        "parallelism TP/PP/DP",
        "DeepSpeed ZeRO stage 0-3",
        "training.total_tokens_b for llm-analysis time",
        "monthly rental cost calculation",
    ]
    approximated: list[str] = [
        "llm-analysis time is a theoretical estimator adjusted by configured efficiency",
        "gpu-mem-calculator memory is an estimate; activation and framework overhead can vary",
    ]
    unsupported: list[str] = []
    warnings: list[str] = []

    if config.model.architecture == "moe":
        supported.append("MoE schema fields and basic expert parameters")
        approximated.append(
            "MoE memory/time support is passed through to backends and should be benchmarked"
        )

    if config.training.mode in {"lora", "qlora"} or config.peft.enabled:
        unsupported.extend(
            [
                "PEFT/LoRA/QLoRA precise memory estimate",
                "PEFT/LoRA/QLoRA precise training time estimate",
            ]
        )
        warnings.append("PEFT is schema-only in this MVP; adapters do not claim precise support")

    if config.deepspeed.offload_optimizer != "none" or config.deepspeed.offload_param != "none":
        supported.append("gpu-mem-calculator DeepSpeed offload memory estimate")
        approximated.append("llm-analysis time adapter does not model CPU/NVMe offload overhead")

    if config.training.precision in {"fp32", "int8", "int4"}:
        approximated.append(
            "precision mapping outside fp16/bf16 may be backend-specific or unsupported for training"
        )

    return CoverageResult(
        supported=supported,
        approximated=approximated,
        unsupported=unsupported,
        warnings=warnings,
    )

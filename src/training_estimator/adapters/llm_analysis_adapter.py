from __future__ import annotations

import logging
import sys
from pathlib import Path

from training_estimator.schema import EstimatorConfig, TimeBackendResult


def _ensure_llm_analysis_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_path = repo_root / "third_party" / "llm-analysis"
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))


def _gpu_config_name(config: EstimatorConfig) -> str:
    gpu_type = config.hardware.gpu_type.lower()
    if "h100" in gpu_type and int(config.hardware.gpu_memory_gb) == 80:
        return "h100-sxm-80gb"
    if "a100" in gpu_type and int(config.hardware.gpu_memory_gb) == 80:
        return "a100-sxm-80gb"
    if "a100" in gpu_type and int(config.hardware.gpu_memory_gb) == 40:
        return "a100-sxm-40gb"
    return config.hardware.profile


def _dtype_name(config: EstimatorConfig) -> str | None:
    if config.training.precision in {"fp16", "bf16"}:
        return "w16a16e16"
    if config.training.precision == "int8":
        return "w8a8e16"
    if config.training.precision == "int4":
        return "w4a16e16"
    return None


class LLMAnalysisAdapter:
    """Normalize llm-analysis training-time results for the orchestrator pipeline."""

    backend = "llm-analysis"

    def estimate_time(self, config: EstimatorConfig) -> TimeBackendResult:
        if config.training.mode != "full":
            return TimeBackendResult(
                backend=self.backend,
                status="unsupported",
                parsed=False,
                warnings=[
                    "PEFT/LoRA/QLoRA time is schema-only in this MVP; llm-analysis models full fine-tuning"
                ],
            )

        dtype_name = _dtype_name(config)
        if dtype_name is None:
            return TimeBackendResult(
                backend=self.backend,
                status="unsupported",
                parsed=False,
                warnings=["llm-analysis training adapter supports fp16/bf16/int8/int4 dtype mappings"],
            )

        try:
            _ensure_llm_analysis_path()
            logging.getLogger("__name__").setLevel(logging.ERROR)
            from llm_analysis.analysis import ActivationRecomputation, DSZeRO, LLMAnalysis
            from llm_analysis.config import (
                ModelConfig,
                ParallelismConfig,
                get_dtype_config_by_name,
                get_gpu_config_by_name,
            )
            from llm_analysis.logger import logger
        except Exception as exc:
            return TimeBackendResult(
                backend=self.backend,
                status="error",
                parsed=False,
                warnings=[f"failed to import llm-analysis: {exc}"],
            )

        try:
            logger.setLevel(logging.ERROR)
            model = ModelConfig(
                name=config.model.name,
                num_layers=config.model.num_layers,
                n_head=config.model.num_attention_heads,
                hidden_dim=config.model.hidden_size,
                vocab_size=config.model.vocab_size,
                max_seq_len=config.model.seq_len,
                moe_num_experts=config.model.moe.num_experts or 1,
                moe_top_k=config.model.moe.experts_per_token or 1,
            )
            gpu = get_gpu_config_by_name(_gpu_config_name(config))
            dtype = get_dtype_config_by_name(dtype_name)
            parallelism = ParallelismConfig(
                tp_size=config.parallelism.tensor_parallel_size,
                pp_size=config.parallelism.pipeline_parallel_size,
                dp_size=config.parallelism.data_parallel_size,
                ep_size=config.parallelism.expert_parallel_size,
                sp_size=(
                    config.parallelism.tensor_parallel_size
                    if config.parallelism.sequence_parallel
                    else 1
                ),
            )
            analysis = LLMAnalysis(
                model_config=model,
                gpu_config=gpu,
                dtype_config=dtype,
                parallelism_config=parallelism,
                flops_efficiency=config.hardware.flops_efficiency,
                hbm_memory_efficiency=config.hardware.memory_efficiency,
            )
            raw = analysis.training(
                batch_size_per_gpu=config.training.micro_batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                global_batch_size=config.training.global_batch_size,
                seq_len=config.model.seq_len,
                total_num_tokens=int(config.training.total_tokens_b * 1_000_000_000),
                activation_recomputation=ActivationRecomputation(
                    config.training.activation_checkpointing
                ),
                ds_zero=DSZeRO(config.deepspeed.zero_stage if config.deepspeed.enabled else 0),
            )
        except Exception as exc:
            return TimeBackendResult(
                backend=self.backend,
                status="error",
                parsed=False,
                warnings=[f"llm-analysis failed: {exc}"],
            )

        total_seconds = raw.get("total_training_latency")
        total_hours = total_seconds / 3600 if total_seconds is not None else None
        estimate_type = (
            "theoretical_lower_bound"
            if config.hardware.flops_efficiency == 1 and config.hardware.memory_efficiency == 1
            else "adjusted_by_efficiency"
        )
        warnings = [
            "llm-analysis is a theoretical estimator; calibrate efficiencies with benchmarks"
        ]
        if config.model.architecture == "moe":
            warnings.append("MoE time estimate should be treated as approximate")
        if config.deepspeed.offload_optimizer != "none" or config.deepspeed.offload_param != "none":
            warnings.append("CPU/NVMe offload overhead is not modeled in llm-analysis time")

        device_tps = raw.get("device_tokens_per_sec")
        tokens_per_second = (
            device_tps * config.hardware.gpu_count if isinstance(device_tps, (int, float)) else None
        )

        return TimeBackendResult(
            backend=self.backend,
            status="ok",
            estimate_type=estimate_type,
            total_time_hours=total_hours,
            gpu_hours=raw.get("gpu_hours"),
            tokens_per_second=tokens_per_second,
            step_time_sec=raw.get("latency_per_iter"),
            raw_output=raw,
            parsed=True,
            warnings=warnings,
        )

from __future__ import annotations

import sys
from pathlib import Path

from training_estimator.schema import EstimatorConfig, MemoryBackendResult


def _ensure_gpu_mem_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_path = repo_root / "third_party" / "gpu-mem-calculator" / "src"
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))


class GPUMemCalculatorAdapter:
    """Normalize gpu-mem-calculator results for the orchestrator pipeline."""

    backend = "gpu-mem-calculator"

    def estimate_memory(self, config: EstimatorConfig) -> MemoryBackendResult:
        if config.training.mode != "full":
            return MemoryBackendResult(
                backend=self.backend,
                status="unsupported",
                fits=None,
                gpu_memory_limit_gb=config.hardware.gpu_memory_gb,
                warnings=[
                    "PEFT/LoRA/QLoRA memory is schema-only in this MVP; no precise estimate emitted"
                ],
            )

        try:
            _ensure_gpu_mem_path()
            from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
            from gpu_mem_calculator.core.models import (
                EngineConfig,
                GPUConfig,
                ModelConfig,
                ParallelismConfig,
                TrainingConfig,
            )
        except Exception as exc:
            return MemoryBackendResult(
                backend=self.backend,
                status="error",
                gpu_memory_limit_gb=config.hardware.gpu_memory_gb,
                warnings=[f"failed to import gpu-mem-calculator: {exc}"],
            )

        try:
            model = ModelConfig(
                name=config.model.name,
                num_parameters=int(config.model.total_params_b * 1_000_000_000),
                num_layers=config.model.num_layers,
                hidden_size=config.model.hidden_size,
                num_attention_heads=config.model.num_attention_heads,
                vocab_size=config.model.vocab_size,
                max_seq_len=config.model.seq_len,
                moe_enabled=config.model.architecture == "moe",
                num_experts=config.model.moe.num_experts or 1,
                top_k=config.model.moe.experts_per_token or 1,
            )
            training = TrainingConfig(
                batch_size=config.training.micro_batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                optimizer=config.training.optimizer,
                dtype=config.training.precision,
                activation_checkpointing=config.training.activation_checkpointing,
            )
            parallelism = ParallelismConfig(
                tensor_parallel_size=config.parallelism.tensor_parallel_size,
                pipeline_parallel_size=config.parallelism.pipeline_parallel_size,
                data_parallel_size=config.parallelism.data_parallel_size,
                sequence_parallel=config.parallelism.sequence_parallel,
            )
            engine = EngineConfig(
                type="deepspeed" if config.deepspeed.enabled else "pytorch_ddp",
                zero_stage=config.deepspeed.zero_stage if config.deepspeed.enabled else 0,
                offload_optimizer=config.deepspeed.offload_optimizer,
                offload_param=config.deepspeed.offload_param,
            )
            gpu = GPUConfig(
                num_gpus=config.hardware.gpu_count,
                gpu_memory_gb=config.hardware.gpu_memory_gb,
            )

            result = GPUMemoryCalculator(
                model_config=model,
                training_config=training,
                parallelism_config=parallelism,
                engine_config=engine,
                gpu_config=gpu,
            ).calculate()
        except Exception as exc:
            return MemoryBackendResult(
                backend=self.backend,
                status="error",
                gpu_memory_limit_gb=config.hardware.gpu_memory_gb,
                warnings=[f"gpu-mem-calculator failed: {exc}"],
            )

        warnings: list[str] = []
        if config.model.architecture == "moe":
            warnings.append("MoE result is backend-dependent; validate against real profile data")
        if config.training.precision in {"int8", "int4"}:
            warnings.append("low-bit training memory support is backend-specific")

        return MemoryBackendResult(
            backend=self.backend,
            status="ok",
            fits=result.fits_on_gpu,
            memory_per_gpu_gb=result.total_memory_per_gpu_gb,
            gpu_memory_limit_gb=config.hardware.gpu_memory_gb,
            memory_utilization=result.memory_utilization_percent / 100,
            breakdown={
                "parameters_gb": result.breakdown.model_params_gb,
                "gradients_gb": result.breakdown.gradients_gb,
                "optimizer_states_gb": result.breakdown.optimizer_states_gb,
                "activations_gb": result.breakdown.activations_gb,
                "reserved_gb": result.breakdown.overhead_gb,
            },
            raw_output=result.model_dump(mode="json"),
            warnings=warnings,
        )

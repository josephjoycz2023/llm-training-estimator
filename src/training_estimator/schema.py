from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


Architecture = Literal["dense", "moe"]
TrainingMode = Literal["full", "lora", "qlora"]
Precision = Literal["fp32", "fp16", "bf16", "int8", "int4"]
Optimizer = Literal["adamw", "adamw_8bit", "sgd"]
OffloadDevice = Literal["none", "cpu", "nvme"]
RiskLevel = Literal["low", "medium", "high", "unknown"]


class RunConfig(BaseModel):
    name: str
    description: str | None = None


class MoEConfig(BaseModel):
    num_experts: int | None = None
    experts_per_token: int | None = None
    expert_parallel_size: int = 1


class ModelConfig(BaseModel):
    name: str
    architecture: Architecture = "dense"
    total_params_b: float = Field(gt=0)
    active_params_b: float | None = Field(default=None, gt=0)
    num_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    seq_len: int = Field(gt=0)
    moe: MoEConfig = Field(default_factory=MoEConfig)

    @model_validator(mode="after")
    def validate_architecture(self) -> "ModelConfig":
        if self.architecture == "dense":
            if self.active_params_b is None:
                self.active_params_b = self.total_params_b
            if self.moe.num_experts is not None or self.moe.experts_per_token is not None:
                raise ValueError("dense models must leave moe.num_experts and moe.experts_per_token empty")
            if self.moe.expert_parallel_size != 1:
                raise ValueError("dense models must use moe.expert_parallel_size = 1")
            return self

        if self.active_params_b is None:
            raise ValueError("moe models require model.active_params_b")
        if self.moe.num_experts is None:
            raise ValueError("moe models require model.moe.num_experts")
        if self.moe.experts_per_token is None:
            raise ValueError("moe models require model.moe.experts_per_token")
        if self.moe.expert_parallel_size < 1:
            raise ValueError("moe expert_parallel_size must be >= 1")
        return self


class TrainingConfig(BaseModel):
    mode: TrainingMode = "full"
    precision: Precision = "bf16"
    optimizer: Optimizer = "adamw"
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    global_batch_size: int = Field(gt=0)
    epochs: int = Field(default=1, gt=0)
    total_tokens_b: float = Field(gt=0)
    activation_checkpointing: int = Field(default=0, ge=0, le=4)


class PeftConfig(BaseModel):
    enabled: bool = False
    method: Literal["lora", "qlora"] | None = None
    trainable_params_b: float | None = Field(default=None, gt=0)
    lora_rank: int | None = Field(default=None, gt=0)
    lora_alpha: int | None = Field(default=None, gt=0)
    target_modules: list[str] | None = None


class ParallelismConfig(BaseModel):
    num_gpus: int = Field(gt=0)
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    expert_parallel_size: int = Field(default=1, ge=1)
    sequence_parallel: bool = False


class DeepSpeedConfig(BaseModel):
    enabled: bool = False
    zero_stage: int = Field(default=0, ge=0, le=3)
    offload_optimizer: OffloadDevice = "none"
    offload_param: OffloadDevice = "none"


class HardwareConfig(BaseModel):
    profile: str
    gpu_type: str
    gpu_count: int = Field(gt=0)
    gpu_memory_gb: float = Field(gt=0)
    flops_efficiency: float = Field(default=1.0, gt=0, le=1)
    memory_efficiency: float = Field(default=1.0, gt=0, le=1)


class CostConfig(BaseModel):
    price_profile: str


class EstimatorConfig(BaseModel):
    version: str = "0.1"
    run: RunConfig
    model: ModelConfig
    training: TrainingConfig
    peft: PeftConfig = Field(default_factory=PeftConfig)
    parallelism: ParallelismConfig
    deepspeed: DeepSpeedConfig = Field(default_factory=DeepSpeedConfig)
    hardware: HardwareConfig
    cost: CostConfig

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "EstimatorConfig":
        expected_gpus = (
            self.parallelism.tensor_parallel_size
            * self.parallelism.pipeline_parallel_size
            * self.parallelism.data_parallel_size
        )
        if expected_gpus != self.parallelism.num_gpus:
            raise ValueError(
                "parallelism.num_gpus must equal tensor_parallel_size * "
                "pipeline_parallel_size * data_parallel_size"
            )
        if self.hardware.gpu_count != self.parallelism.num_gpus:
            raise ValueError("hardware.gpu_count must equal parallelism.num_gpus")

        if self.model.architecture == "dense" and self.parallelism.expert_parallel_size != 1:
            raise ValueError("dense models must use parallelism.expert_parallel_size = 1")
        if self.model.architecture == "moe" and self.parallelism.expert_parallel_size < 1:
            raise ValueError("moe models require parallelism.expert_parallel_size >= 1")

        if self.training.mode == "full":
            if self.peft.enabled:
                raise ValueError("training.mode = full requires peft.enabled = false")
        else:
            if not self.peft.enabled:
                raise ValueError("training.mode = lora/qlora requires peft.enabled = true")
            if self.peft.method != self.training.mode:
                raise ValueError("peft.method must match training.mode for PEFT training")
            if self.peft.trainable_params_b is None:
                raise ValueError("PEFT modes require peft.trainable_params_b in this MVP")
        return self


class MemoryBackendResult(BaseModel):
    backend: str = "gpu-mem-calculator"
    status: Literal["ok", "unsupported", "error"] = "ok"
    fits: bool | None = None
    memory_per_gpu_gb: float | None = None
    gpu_memory_limit_gb: float | None = None
    memory_utilization: float | None = None
    breakdown: dict[str, float | None] = Field(default_factory=dict)
    raw_output: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class TimeBackendResult(BaseModel):
    backend: str = "llm-analysis"
    status: Literal["ok", "unsupported", "error", "not_actionable"] = "ok"
    estimate_type: str = "adjusted_by_efficiency"
    total_time_hours: float | None = None
    gpu_hours: float | None = None
    tokens_per_second: float | None = None
    step_time_sec: float | None = None
    raw_output: dict[str, Any] | None = None
    parsed: bool = True
    warnings: list[str] = Field(default_factory=list)


class CostResult(BaseModel):
    currency: str
    price_profile: str
    monthly_rental_cost: float
    hours_per_month: float
    estimated_cost: float
    warnings: list[str] = Field(default_factory=list)


class CoverageResult(BaseModel):
    supported: list[str] = Field(default_factory=list)
    approximated: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FinalResult(BaseModel):
    run_name: str
    runnable: bool
    risk_level: RiskLevel
    memory: MemoryBackendResult
    time: TimeBackendResult | None = None
    cost: CostResult | None = None
    coverage: CoverageResult
    warnings: list[str] = Field(default_factory=list)

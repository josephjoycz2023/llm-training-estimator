# YAML Schema Reference

This document describes every tunable field in the estimator YAML files.

There are two YAML inputs:

- Main run config: passed to `lte validate` and `lte estimate`.
- Price config: passed to `lte estimate --prices`.

The implementation source of truth is `src/training_estimator/schema.py` and
`src/training_estimator/config_loader.py`.

## Main Run Config

Top-level structure:

```yaml
version: "0.1"
run: {}
model: {}
training: {}
peft: {}
parallelism: {}
deepspeed: {}
hardware: {}
cost: {}
```

### `version`

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `version` | string | no | `"0.1"` | any string | Config schema version marker. Currently informational. |

### `run`

Metadata for the estimate.

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `run.name` | string | yes | none | non-empty string recommended | Run identifier used in reports and output JSON. |
| `run.description` | string or null | no | `null` | any string | Human-readable note for the run. |

Example:

```yaml
run:
  name: qwen_7b_full_h100_zero3
  description: qwen-like dense full fine-tuning on H100 x8
```

### `model`

Model architecture and scale.

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `model.name` | string | yes | none | any string | Model name passed through to backend configs. |
| `model.architecture` | string | no | `dense` | `dense`, `moe` | Selects dense or Mixture-of-Experts validation behavior. |
| `model.total_params_b` | float | yes | none | `> 0` | Total model parameters in billions. Used by memory adapter. |
| `model.active_params_b` | float or null | conditional | `null` | `> 0` when set | Active parameters in billions. Dense defaults to `total_params_b`; MoE requires this. |
| `model.num_layers` | integer | yes | none | `> 0` | Transformer layer count. |
| `model.hidden_size` | integer | yes | none | `> 0` | Hidden dimension. |
| `model.num_attention_heads` | integer | yes | none | `> 0` | Attention head count. |
| `model.vocab_size` | integer | yes | none | `> 0` | Vocabulary size. |
| `model.seq_len` | integer | yes | none | `> 0` | Training sequence length. |
| `model.moe` | mapping | no | see below | required for MoE details | MoE-specific parameters. |

#### `model.moe`

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `model.moe.num_experts` | integer or null | conditional | `null` | required for `architecture=moe`; must be null for dense | Total number of experts. |
| `model.moe.experts_per_token` | integer or null | conditional | `null` | required for `architecture=moe`; must be null for dense | Top-k experts activated per token. |
| `model.moe.expert_parallel_size` | integer | no | `1` | `>= 1`; must be `1` for dense | Model-local EP setting. Kept for schema clarity; pipeline also uses `parallelism.expert_parallel_size`. |

Dense validation:

- `active_params_b` may be `null`; it is filled as `total_params_b`.
- `model.moe.num_experts` must be `null`.
- `model.moe.experts_per_token` must be `null`.
- `model.moe.expert_parallel_size` must be `1`.
- `parallelism.expert_parallel_size` must be `1`.

MoE validation:

- `active_params_b` is required.
- `model.moe.num_experts` is required.
- `model.moe.experts_per_token` is required.
- Expert parallel sizes must be `>= 1`.
- MVP behavior: MoE is passed through to backends, but reports include an approximation warning.

### `training`

Training mode, precision, batch shape, token count, and checkpointing.

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `training.mode` | string | no | `full` | `full`, `lora`, `qlora` | Training method. Only `full` has precise backend estimates in this MVP. |
| `training.precision` | string | no | `bf16` | `fp32`, `fp16`, `bf16`, `int8`, `int4` | Training precision. Backend support varies for low-bit modes. |
| `training.optimizer` | string | no | `adamw` | `adamw`, `adamw_8bit`, `sgd` | Optimizer passed to gpu-mem-calculator. |
| `training.micro_batch_size` | integer | yes | none | `> 0` | Per-GPU micro batch size. |
| `training.gradient_accumulation_steps` | integer | yes | none | `> 0` | Number of accumulation steps. |
| `training.global_batch_size` | integer | yes | none | `> 0` | Global batch size used by llm-analysis. |
| `training.epochs` | integer | no | `1` | `> 0` | Informational in the current MVP; token count drives time estimate. |
| `training.total_tokens_b` | float | yes | none | `> 0` | Total training tokens in billions. Used by llm-analysis time estimate. |
| `training.activation_checkpointing` | integer | no | `0` | `0`, `1`, `2`, `3`, `4` | Activation recomputation level passed to both backends. |

`training.activation_checkpointing` maps to llm-analysis recomputation levels:

| Value | Meaning |
| --- | --- |
| `0` | no activation recomputation |
| `1` | attention compute recomputation |
| `2` | attention recomputation |
| `3` | norm-attention-norm recomputation |
| `4` | full activation recomputation |

PEFT behavior:

- `training.mode=full` requires `peft.enabled=false`.
- `training.mode=lora` or `qlora` requires `peft.enabled=true`.
- `training.mode=lora` or `qlora` requires `peft.method` to match the mode.
- `training.mode=lora` or `qlora` requires `peft.trainable_params_b`.
- MVP behavior: PEFT validates and appears in coverage warnings, but adapters do not emit precise memory/time estimates.

### `peft`

PEFT metadata. This block is intentionally present even though precise PEFT formulas are outside the MVP.

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `peft.enabled` | boolean | no | `false` | `true`, `false` | Enables PEFT schema path. Must be false for full fine-tuning. |
| `peft.method` | string or null | conditional | `null` | `lora`, `qlora`, or `null` | Must match `training.mode` when mode is `lora` or `qlora`. |
| `peft.trainable_params_b` | float or null | conditional | `null` | `> 0` when set | Trainable parameter count in billions. Required for PEFT modes. |
| `peft.lora_rank` | integer or null | no | `null` | `> 0` when set | LoRA rank metadata. |
| `peft.lora_alpha` | integer or null | no | `null` | `> 0` when set | LoRA alpha metadata. |
| `peft.target_modules` | list of strings or null | no | `null` | list values should be module names | Target module names for LoRA metadata. |

### `parallelism`

Parallelism layout.

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `parallelism.num_gpus` | integer | yes | none | `> 0` | Total GPU count used by the run. |
| `parallelism.tensor_parallel_size` | integer | no | `1` | `>= 1` | Tensor parallel degree. |
| `parallelism.pipeline_parallel_size` | integer | no | `1` | `>= 1` | Pipeline parallel degree. |
| `parallelism.data_parallel_size` | integer | no | `1` | `>= 1` | Data parallel degree. |
| `parallelism.expert_parallel_size` | integer | no | `1` | `>= 1`; must be `1` for dense | Expert parallel degree passed to llm-analysis. |
| `parallelism.sequence_parallel` | boolean | no | `false` | `true`, `false` | Enables sequence parallel mapping in llm-analysis. |

Cross-field rule:

```text
parallelism.num_gpus
  = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
```

Also:

```text
hardware.gpu_count = parallelism.num_gpus
```

### `deepspeed`

DeepSpeed ZeRO and offload settings.

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `deepspeed.enabled` | boolean | no | `false` | `true`, `false` | Selects DeepSpeed engine in gpu-mem-calculator. |
| `deepspeed.zero_stage` | integer | no | `0` | `0`, `1`, `2`, `3` | ZeRO stage passed to both backends when enabled. |
| `deepspeed.offload_optimizer` | string | no | `none` | `none`, `cpu`, `nvme` | Optimizer state offload setting. Memory adapter uses it; time adapter warns that offload overhead is not modeled. |
| `deepspeed.offload_param` | string | no | `none` | `none`, `cpu`, `nvme` | Parameter offload setting. Memory adapter uses it; time adapter warns that offload overhead is not modeled. |

If `deepspeed.enabled=false`, the memory adapter uses `pytorch_ddp`, and the time adapter uses ZeRO stage `0`.

### `hardware`

Hardware profile and efficiency assumptions.

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `hardware.profile` | string | yes | none | any string | Hardware profile name. Also used as fallback for llm-analysis GPU config lookup. |
| `hardware.gpu_type` | string | yes | none | examples: `H100-80G`, `A100-80G` | Human-readable GPU type. Adapter maps common H100/A100 names to llm-analysis presets. |
| `hardware.gpu_count` | integer | yes | none | `> 0`; must equal `parallelism.num_gpus` | Total GPU count. |
| `hardware.gpu_memory_gb` | float | yes | none | `> 0` | Memory per GPU in GB. |
| `hardware.flops_efficiency` | float | no | `1.0` | `0 < value <= 1` | Achieved FLOPS efficiency passed to llm-analysis. Use benchmark-calibrated values when possible. |
| `hardware.memory_efficiency` | float | no | `1.0` | `0 < value <= 1` | HBM memory efficiency passed to llm-analysis. |

Current built-in adapter mapping:

| `gpu_type` / memory | llm-analysis GPU config |
| --- | --- |
| contains `H100` and `gpu_memory_gb=80` | `h100-sxm-80gb` |
| contains `A100` and `gpu_memory_gb=80` | `a100-sxm-80gb` |
| contains `A100` and `gpu_memory_gb=40` | `a100-sxm-40gb` |
| otherwise | `hardware.profile` |

### `cost`

Cost profile selection.

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `cost.price_profile` | string | yes | none | must exist in price YAML `hardware_prices` | Selects monthly rental price profile. |

Cost is only emitted when:

- memory backend status is `ok`,
- `memory.fits=true`,
- time backend status is `ok`,
- `time.total_time_hours` is available.

Cost formula:

```text
hourly_cluster_cost = monthly_rental_cost / hours_per_month
estimated_cost = total_time_hours * hourly_cluster_cost
```

If the price profile GPU count differs from `hardware.gpu_count`, the MVP scales by:

```text
scaled_monthly_cost = monthly_rental_cost * hardware.gpu_count / price_gpu_count
```

## Price Config

Example:

```yaml
currency: CNY
hours_per_month: 720

hardware_prices:
  h100_80g_8gpu:
    gpu_type: H100-80G
    gpu_count: 8
    monthly_rental_cost: 45000
```

Top-level fields:

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `currency` | string | no | `CNY` | any string | Currency label used in reports. |
| `hours_per_month` | float | no | `720` | `> 0` | Divisor for converting monthly rental to hourly cluster cost. |
| `hardware_prices` | mapping | yes | none | keys are profile names | Price profile table. |

Each `hardware_prices.<profile>` item:

| Field | Type | Required | Default | Allowed / Constraint | Meaning |
| --- | --- | --- | --- | --- | --- |
| `gpu_type` | string | yes | none | any string | Human-readable GPU type for the price profile. |
| `gpu_count` | integer | yes | none | `> 0` | Number of GPUs covered by this monthly rental price. |
| `monthly_rental_cost` | float | yes | none | `>= 0` | Monthly rental cost for the profile. |

Backward compatibility:

- The loader also accepts old price files using top-level `hardware:` instead of `hardware_prices:`.

## Minimal Dense Example

```yaml
version: "0.1"

run:
  name: qwen_7b_full_h100_zero3
  description: qwen-like dense full fine-tuning on H100 x8

model:
  name: qwen-7b-like
  architecture: dense
  total_params_b: 7
  active_params_b: null
  num_layers: 32
  hidden_size: 4096
  num_attention_heads: 32
  vocab_size: 152064
  seq_len: 4096
  moe:
    num_experts: null
    experts_per_token: null
    expert_parallel_size: 1

training:
  mode: full
  precision: bf16
  optimizer: adamw
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  global_batch_size: 64
  epochs: 1
  total_tokens_b: 1
  activation_checkpointing: 1

peft:
  enabled: false
  method: null
  trainable_params_b: null
  lora_rank: null
  lora_alpha: null
  target_modules: null

parallelism:
  num_gpus: 8
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  data_parallel_size: 8
  expert_parallel_size: 1
  sequence_parallel: false

deepspeed:
  enabled: true
  zero_stage: 3
  offload_optimizer: none
  offload_param: none

hardware:
  profile: h100_80g_8gpu
  gpu_type: H100-80G
  gpu_count: 8
  gpu_memory_gb: 80
  flops_efficiency: 0.35
  memory_efficiency: 0.90

cost:
  price_profile: h100_80g_8gpu
```

## MVP Support Notes

- Dense full fine-tuning is the primary supported path.
- MoE validates and is passed through, but reports approximation warnings.
- PEFT validates and appears in coverage warnings, but precise memory/time estimates are not claimed.
- Offload is used by the memory adapter; llm-analysis time does not model offload overhead.
- Efficiency values should be treated as calibration knobs, not universal constants.

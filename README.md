# LLM Training Estimator

This project is an MVP orchestrator for LLM training estimates. It does not replace
`llm-analysis` or `gpu-mem-calculator`; it loads one unified YAML config, calls both
backends, normalizes their results, adds coverage warnings, and computes rental cost.

## Pipeline

```text
YAML config
  -> Pydantic schema validation
  -> derived/default fields
  -> gpu-mem-calculator input
  -> llm-analysis input
  -> normalized backend results
  -> memory risk and runnable decision
  -> rental cost calculation
  -> table / json / markdown report
```

The pipeline always runs memory first. If the config does not fit in GPU memory,
`runnable=false`, `time.status=not_actionable`, and no valid cost is emitted.

## Install

```bash
python -m pip install -e .
```

If `lte` is not on your PATH, run the CLI as:

```bash
python -m training_estimator.cli --help
```

## Commands

Validate a config:

```bash
python -m training_estimator.cli validate configs/example_qwen_7b_zero3.yaml
```

Run the full estimate:

```bash
python -m training_estimator.cli estimate configs/example_qwen_7b_zero3.yaml \
  --prices configs/hardware_prices.yaml \
  --format table
```

Write JSON output:

```bash
python -m training_estimator.cli estimate configs/example_qwen_7b_zero3.yaml \
  --prices configs/hardware_prices.yaml \
  --format json \
  --output outputs/qwen_7b_full_h100_zero3.json
```

Supported formats are `table`, `json`, and `markdown`.

## Web UI

The Web UI follows the Astro setup used by `Personalized-Research-Dashboard/web`.
It provides block-by-block schema inputs, typed validation, English/Chinese meaning
tooltips, estimator results, and an optional Markdown summary from OpenAI or DeepSeek.

Start the Python API:

```bash
python -m uvicorn training_estimator.web_api:app --reload --port 8000
```

Start the Astro UI in another terminal:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:4321`. The summary API reads local test keys from the same
places as the research dashboard: `D:/paper_daily/Personalized-Research-Dashboard/config.yaml`,
its sibling `config.local.yaml`, this project's own `config.local.yaml`, or
`OPENAI_API_KEY` / `DEEPSEEK_API_KEY`.

## Unified Config

The MVP config is intentionally strict and split into these blocks:

- `run`: name and description.
- `model`: architecture, parameter scale, layers, hidden size, heads, vocab, sequence length, MoE fields.
- `training`: full/LoRA/QLoRA mode, precision, optimizer, batch sizes, tokens, activation checkpointing.
- `peft`: PEFT metadata. In this MVP it is schema and coverage only.
- `parallelism`: GPU count, TP, PP, DP, EP, sequence parallel.
- `deepspeed`: ZeRO stage and offload settings.
- `hardware`: GPU type/count/memory and efficiency assumptions.
- `cost`: price profile key.

Full field-level schema reference: [docs/yaml_schema.md](docs/yaml_schema.md).

See:

- `configs/example_qwen_7b_zero3.yaml`
- `configs/example_moe_full.yaml`
- `configs/hardware_prices.yaml`

## Backend Responsibilities

`gpu-mem-calculator` estimates:

- fits / OOM
- memory per GPU
- memory utilization
- parameter, gradient, optimizer, activation, and reserved memory breakdown

`llm-analysis` estimates:

- theoretical or efficiency-adjusted training time
- GPU-hours
- step time and rough tokens/sec when available

`training_estimator.cost` calculates:

```text
hourly_cluster_cost = monthly_rental_cost / hours_per_month
estimated_cost = total_time_hours * hourly_cluster_cost
```

## Risk Levels

Memory risk is derived from `memory_utilization`:

- `< 80%`: `low`
- `80% - 92%`: `medium`
- `>= 92%`: `high`
- OOM: `high`

## Current MVP Boundary

Supported first:

- dense full fine-tuning
- H100/A100 80GB 8-GPU profiles
- DeepSpeed ZeRO 0/1/2/3
- monthly rental cost calculation
- table, JSON, and Markdown reports

Handled with warnings:

- MoE is accepted by schema and passed to both backends, but results should be benchmarked.
- CPU/NVMe offload affects memory estimates; time overhead is not modeled.
- `llm-analysis` output is theoretical and depends on configured efficiency.

Not claimed as precise:

- LoRA / QLoRA / PEFT memory and time. These modes validate through schema and appear in
  coverage warnings, but adapters return unsupported instead of pretending to be exact.

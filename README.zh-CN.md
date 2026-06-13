[English](README.md) | [简体中文](README.zh-CN.md)

# LLM Training Estimator

LLM Training Estimator 是一个面向大模型训练规划的统一工具集，
将显存估算、训练时长估算、租赁成本计算、硬件横向对比，以及可选的
LLM 总结分析整合到同一条工作流中。

你只需要提供一份结构化 YAML 配置，系统就会调用底层分析引擎，
输出统一格式的结果，便于评审、比较和分享。

## 功能概览

- 使用统一 YAML schema 描述模型、训练、并行、DeepSpeed、硬件与成本参数
- 估算单卡显存、显存利用率、是否可运行，以及显存构成拆分
- 估算训练时长、GPU-hours，以及结合效率参数的运行时间
- 基于硬件价格 profile 计算训练成本
- 给出风险等级与近似/不支持能力的覆盖提示
- 提供带类型校验的 Web UI，支持中英文切换与进度展示
- 支持多硬件预设对比训练时长与成本
- 支持生成可选的 Markdown 总结分析
- 支持 CLI、JSON、表格和 Markdown 输出

## Pipeline 输出内容

对每次估算，系统可以产出：

- `runnable`
- `risk_level`
- 平均单卡显存与显存利用率
- 训练时长估计
- GPU-hours
- 成本估计
- warnings 与 coverage 说明

## 系统流程

```text
YAML 配置
  -> schema 校验
  -> 显存估算
  -> 可运行性与风险判断
  -> 时长估算
  -> 成本计算
  -> 对比 / 总结展示
```

系统总是先进行显存阶段。如果配置本身无法装入显存，结果会被标记为
不可运行，后续阶段会按此状态处理。

## 安装指南

### 环境要求

- Python 3.10+
- Node.js 22.12+（用于 Web UI）
- `git`，并支持 submodule（如果你是自行克隆仓库）

### 1. 克隆仓库

如果仓库中的第三方源码通过 submodule 提供，建议使用：

```bash
git clone --recurse-submodules <your-repo-url>
cd llm-training-estimator
```

如果已经克隆但没有初始化 submodule：

```bash
git submodule update --init --recursive
```

### 2. 安装主项目

```bash
python -m pip install -e .
```

### 3. 安装内置第三方分析后端

本项目集成了两个上游分析引擎。建议把它们作为本地包安装，
这样 estimator 可以直接导入：

```bash
python -m pip install -e third_party/llm-analysis
python -m pip install -e third_party/gpu-mem-calculator
```

如果你希望直接使用上游发布版本，也可以：

```bash
python -m pip install llm-analysis
python -m pip install gpu-mem-calculator
```

### 4. 安装 Web UI

```bash
cd web
npm install
cd ..
```

## 快速开始

校验配置文件：

```bash
python -m training_estimator.cli validate configs/example_qwen_7b_zero3.yaml
```

以表格形式运行估算：

```bash
python -m training_estimator.cli estimate configs/example_qwen_7b_zero3.yaml \
  --prices configs/hardware_prices.yaml \
  --format table
```

输出 JSON 结果：

```bash
python -m training_estimator.cli estimate configs/example_qwen_7b_zero3.yaml \
  --prices configs/hardware_prices.yaml \
  --format json \
  --output outputs/qwen_7b_full_h100_zero3.json
```

支持的输出格式：

- `table`
- `json`
- `markdown`

## Web UI

启动 API 服务：

```bash
python -m uvicorn --app-dir src training_estimator.web_api:app --reload --port 8000
```

在另一个终端启动前端：

```bash
cd web
npm run dev
```

然后打开：

```text
http://localhost:4321
```

Web 界面支持：

- 分块编辑 schema
- 按字段类型进行输入校验
- 中英文界面切换
- 进度与预计耗时展示
- 结果优先返回
- 可选的总结分析
- 多硬件预设对比

## 配置说明

主配置由以下部分组成：

- `run`
- `model`
- `training`
- `peft`
- `parallelism`
- `deepspeed`
- `hardware`
- `cost`

字段级 schema 说明文档：

- [docs/yaml_schema.md](docs/yaml_schema.md)

示例配置：

- `configs/example_qwen_7b_zero3.yaml`
- `configs/example_moe_full.yaml`
- `configs/hardware_prices.yaml`

## 硬件对比

Web UI 支持一次选择多个硬件预设进行横向对比。
系统会在保持模型与训练假设一致的前提下，对每个 GPU 预设分别执行估算，
并输出以下维度的并排结果：

- 训练时长
- 估算成本
- 是否可运行
- 风险等级
- 平均单卡显存

这对于评估“更快的 GPU 是否值得更高成本”，或者“更便宜的集群会不会显著拖慢交付时间”很有帮助。

## 可选总结分析

在核心估算完成后，API 可以继续调用支持的 LLM 提供方生成一段简短的
Markdown 总结，用于快速说明：

- 当前训练方案是否可执行
- 显存压力与 OOM 风险
- 时长与 GPU-hours 的含义
- 成本解释
- 注意事项与后续建议

## 输出结果

常见输出形式包括：

- `outputs/` 下的 JSON 结果
- `outputs/` 下的 Markdown 报告
- CLI 的终端表格输出

## 参考资料

本项目建立在以下第三方分析引擎之上：

1. [llm-analysis](https://github.com/cli99/llm-analysis)  
   用于 Transformer 训练与推理的时延和显存分析。

2. [gpu-mem-calculator](https://github.com/George614/gpu-mem-calculator)  
   用于多种分布式训练策略下的大模型显存估算。

其他参考：

- [DeepSpeed Memory Documentation](https://deepspeed.readthedocs.io/en/latest/memory.html)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)


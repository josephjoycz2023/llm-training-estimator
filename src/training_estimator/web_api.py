from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from training_estimator.config_loader import PriceBook, load_prices
from training_estimator.llm_summary import (
    Language,
    SummaryConfigError,
    SummaryRequestError,
    generate_summary,
)
from training_estimator.pipeline import run_pipeline
from training_estimator.schema import EstimatorConfig, FinalResult


DEFAULT_PRICE_PATH = Path("configs/hardware_prices.yaml")


class EstimateRequest(BaseModel):
    config: dict[str, Any]
    prices: dict[str, Any] | None = None
    provider: Literal["auto", "openai", "deepseek"] = "auto"
    language: Language = "en"
    result: dict[str, Any] | None = None


class CompareRequest(BaseModel):
    config: dict[str, Any]
    prices: dict[str, Any] | None = None
    compare_profiles: list[str]


app = FastAPI(title="LLM Training Estimator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_price_book(prices_payload: dict[str, Any] | None) -> PriceBook:
    if prices_payload is not None:
        return PriceBook.model_validate(prices_payload)
    return load_prices(DEFAULT_PRICE_PATH)


def _infer_gpu_memory_gb(gpu_type: str, fallback: float | None = None) -> float:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:gb|g)", gpu_type.lower())
    if matches:
        return float(matches[-1])
    if fallback is not None:
        return fallback
    raise ValueError(f"unable to infer gpu_memory_gb from gpu_type: {gpu_type}")


def _hardware_option(profile: str, price: Any) -> dict[str, Any]:
    return {
        "profile": profile,
        "gpu_type": price.gpu_type,
        "gpu_count": price.gpu_count,
        "gpu_memory_gb": _infer_gpu_memory_gb(price.gpu_type),
        "monthly_rental_cost": price.monthly_rental_cost,
    }


def _build_compare_config(
    config: EstimatorConfig,
    prices: PriceBook,
    profile: str,
) -> EstimatorConfig:
    price = prices.hardware_prices.get(profile)
    if price is None:
        raise ValueError(f"unknown compare profile: {profile}")

    tp_size = config.parallelism.tensor_parallel_size
    pp_size = config.parallelism.pipeline_parallel_size
    per_replica_gpu_count = tp_size * pp_size
    if price.gpu_count % per_replica_gpu_count != 0:
        raise ValueError(
            f"profile {profile} uses {price.gpu_count} GPUs, which is incompatible with "
            f"tensor_parallel_size={tp_size} and pipeline_parallel_size={pp_size}"
        )

    config_data = config.model_dump(mode="json")
    config_data["hardware"]["profile"] = profile
    config_data["hardware"]["gpu_type"] = price.gpu_type
    config_data["hardware"]["gpu_count"] = price.gpu_count
    config_data["hardware"]["gpu_memory_gb"] = _infer_gpu_memory_gb(
        price.gpu_type,
        config.hardware.gpu_memory_gb,
    )
    config_data["parallelism"]["num_gpus"] = price.gpu_count
    config_data["parallelism"]["data_parallel_size"] = price.gpu_count // per_replica_gpu_count
    config_data["cost"]["price_profile"] = profile
    return EstimatorConfig.model_validate(config_data)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/hardware-options")
def hardware_options() -> dict[str, Any]:
    try:
        prices = load_prices(DEFAULT_PRICE_PATH)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = [
        _hardware_option(profile, price)
        for profile, price in sorted(
            prices.hardware_prices.items(),
            key=lambda item: (item[1].gpu_type.lower(), item[1].gpu_count, item[0]),
        )
    ]
    return {
        "currency": prices.currency,
        "hours_per_month": prices.hours_per_month,
        "items": items,
    }


@app.post("/api/estimate")
def estimate(request: EstimateRequest) -> dict[str, Any]:
    try:
        config = EstimatorConfig.model_validate(request.config)
        prices = _load_price_book(request.prices)
        result = run_pipeline(config, prices)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "result": result.model_dump(mode="json"),
        "summary_markdown": "",
        "summary_provider": None,
        "summary_error": None,
    }


@app.post("/api/summary")
def summary(request: EstimateRequest) -> dict[str, Any]:
    try:
        config = EstimatorConfig.model_validate(request.config)
        if request.result is not None:
            result = FinalResult.model_validate(request.result)
        else:
            prices = _load_price_book(request.prices)
            result = run_pipeline(config, prices)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        summary_markdown, summary_provider = generate_summary(
            config,
            result,
            request.provider,
            request.language,
        )
        summary_error = None
    except (SummaryConfigError, SummaryRequestError) as exc:
        summary_markdown = ""
        summary_provider = None
        summary_error = str(exc)
    except Exception as exc:  # Keep summary failures isolated from estimation success.
        summary_markdown = ""
        summary_provider = None
        summary_error = f"summary generation failed: {exc}"

    return {
        "result": result.model_dump(mode="json"),
        "summary_markdown": summary_markdown,
        "summary_provider": summary_provider,
        "summary_error": summary_error,
    }


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict[str, Any]:
    try:
        config = EstimatorConfig.model_validate(request.config)
        prices = _load_price_book(request.prices)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items: list[dict[str, Any]] = []
    seen_profiles: set[str] = set()
    for profile in request.compare_profiles:
        if profile in seen_profiles:
            continue
        seen_profiles.add(profile)

        price = prices.hardware_prices.get(profile)
        if price is None:
            items.append(
                {
                    "profile": profile,
                    "error": f"unknown compare profile: {profile}",
                    "result": None,
                }
            )
            continue

        option = _hardware_option(profile, price)
        try:
            compare_config = _build_compare_config(config, prices, profile)
            result = run_pipeline(compare_config, prices)
            items.append(
                {
                    **option,
                    "error": None,
                    "result": result.model_dump(mode="json"),
                }
            )
        except Exception as exc:
            items.append(
                {
                    **option,
                    "error": str(exc),
                    "result": None,
                }
            )

    return {
        "base_profile": config.hardware.profile,
        "items": items,
    }

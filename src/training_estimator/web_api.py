from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from training_estimator.config_loader import PriceBook, load_prices
from training_estimator.llm_summary import SummaryConfigError, SummaryRequestError, generate_summary
from training_estimator.pipeline import run_pipeline
from training_estimator.schema import EstimatorConfig, FinalResult


class EstimateRequest(BaseModel):
    config: dict[str, Any]
    prices: dict[str, Any] | None = None
    provider: Literal["auto", "openai", "deepseek"] = "auto"
    result: dict[str, Any] | None = None


app = FastAPI(title="LLM Training Estimator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/estimate")
def estimate(request: EstimateRequest) -> dict[str, Any]:
    try:
        config = EstimatorConfig.model_validate(request.config)
        prices = (
            PriceBook.model_validate(request.prices)
            if request.prices is not None
            else load_prices(Path("configs/hardware_prices.yaml"))
        )
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
            prices = (
                PriceBook.model_validate(request.prices)
                if request.prices is not None
                else load_prices(Path("configs/hardware_prices.yaml"))
            )
            result = run_pipeline(config, prices)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        summary_markdown, summary_provider = generate_summary(config, result, request.provider)
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

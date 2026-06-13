from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from training_estimator.schema import EstimatorConfig


class HardwarePrice(BaseModel):
    gpu_type: str
    gpu_count: int = Field(gt=0)
    monthly_rental_cost: float = Field(ge=0)


class PriceBook(BaseModel):
    currency: str = "CNY"
    hours_per_month: float = Field(default=720, gt=0)
    hardware_prices: dict[str, HardwarePrice]


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_config(path: str | Path) -> EstimatorConfig:
    return EstimatorConfig.model_validate(_load_yaml(path))


def load_prices(path: str | Path) -> PriceBook:
    data = _load_yaml(path)
    if "hardware_prices" not in data and "hardware" in data:
        data["hardware_prices"] = data.pop("hardware")
    return PriceBook.model_validate(data)

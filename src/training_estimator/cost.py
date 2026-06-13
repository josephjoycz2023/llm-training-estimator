from __future__ import annotations

from training_estimator.config_loader import PriceBook
from training_estimator.schema import CostResult, EstimatorConfig, TimeBackendResult


def calculate_cost(
    config: EstimatorConfig,
    time_result: TimeBackendResult,
    prices: PriceBook,
) -> CostResult | None:
    if time_result.total_time_hours is None:
        return None

    price = prices.hardware_prices.get(config.cost.price_profile)
    if price is None:
        raise ValueError(f"unknown price_profile: {config.cost.price_profile}")

    warnings: list[str] = []
    if price.gpu_count != config.hardware.gpu_count:
        warnings.append(
            "price profile GPU count differs from config; cost is scaled by gpu_count ratio"
        )
    scale = config.hardware.gpu_count / price.gpu_count
    monthly_rental_cost = price.monthly_rental_cost * scale
    hourly_cluster_cost = monthly_rental_cost / prices.hours_per_month

    return CostResult(
        currency=prices.currency,
        price_profile=config.cost.price_profile,
        monthly_rental_cost=monthly_rental_cost,
        hours_per_month=prices.hours_per_month,
        estimated_cost=time_result.total_time_hours * hourly_cluster_cost,
        warnings=warnings,
    )

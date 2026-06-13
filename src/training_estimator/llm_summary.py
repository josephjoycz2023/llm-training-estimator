from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import requests
import yaml

from training_estimator.schema import EstimatorConfig, FinalResult


Provider = Literal["auto", "openai", "deepseek"]
Language = Literal["en", "zh"]
LOCAL_CONFIG = Path("config.local.yaml")
REFERENCE_CONFIG = Path("D:/paper_daily/Personalized-Research-Dashboard/config.yaml")


class SummaryConfigError(RuntimeError):
    """Raised when the local LLM config is missing required API settings."""


class SummaryRequestError(RuntimeError):
    """Raised when a provider request fails or returns an unusable response."""


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _normalize_key_candidates(*values: Any) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            for part in [item.strip() for item in re.split(r"[\r\n,;]+", value) if item.strip()]:
                if part not in candidates:
                    candidates.append(part)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)

    for value in values:
        add(value)
    return candidates


def _load_project_local_config() -> dict[str, Any]:
    return _load_yaml_dict(LOCAL_CONFIG)


def load_reference_llm_config(config_path: Path = REFERENCE_CONFIG) -> dict[str, Any]:
    config = _load_yaml_dict(config_path)
    local_path = config_path.with_name("config.local.yaml")
    if local_path.exists():
        config.update(_load_yaml_dict(local_path))
    if LOCAL_CONFIG.exists():
        config.update(_load_project_local_config())

    config["config_file"] = str(config_path)
    config["config_local_path"] = str(LOCAL_CONFIG if LOCAL_CONFIG.exists() else local_path)
    config.setdefault("llm_provider", "openai")
    config.setdefault("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
    config.setdefault("openai_api_keys", os.getenv("OPENAI_API_KEYS", ""))
    config.setdefault("openai_model", os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    config.setdefault("openai_base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    config.setdefault("openai_timeout", int(os.getenv("OPENAI_TIMEOUT", "60")))
    config.setdefault("openai_instructions", "")
    config.setdefault("deepseek_api_key", os.getenv("DEEPSEEK_API_KEY", ""))
    config.setdefault("deepseek_api_keys", os.getenv("DEEPSEEK_API_KEYS", ""))
    config.setdefault("deepseek_model", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    config.setdefault("deepseek_base_url", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    config.setdefault("deepseek_timeout", int(os.getenv("DEEPSEEK_TIMEOUT", "60")))
    config.setdefault("deepseek_instructions", "")
    config.setdefault("deepseek_reasoning_effort", "high")
    config.setdefault("deepseek_thinking_enabled", True)
    config.setdefault("analysis_request_retries", 1)

    config["openai_api_keys"] = _normalize_key_candidates(
        config.get("openai_api_keys", ""),
        config.get("openai_api_key", ""),
    )
    config["deepseek_api_keys"] = _normalize_key_candidates(
        config.get("deepseek_api_keys", ""),
        config.get("deepseek_api_key", ""),
    )
    return config


def _resolve_provider(config: dict[str, Any], requested: Provider) -> Literal["openai", "deepseek"]:
    if requested == "auto":
        requested = str(config.get("llm_provider", "openai")).strip().lower()  # type: ignore[assignment]
    if requested not in {"openai", "deepseek", "auto"}:
        raise SummaryConfigError("provider must be auto, openai, or deepseek")
    if requested == "auto":
        if config.get("deepseek_api_keys"):
            return "deepseek"
        return "openai"
    if requested == "openai" and config.get("openai_api_keys"):
        return "openai"
    if requested == "deepseek" and config.get("deepseek_api_keys"):
        return "deepseek"
    fallback = "deepseek" if requested == "openai" else "openai"
    if config.get(f"{fallback}_api_keys"):
        return fallback  # type: ignore[return-value]
    return requested  # type: ignore[return-value]


def _extract_openai_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(content["text"])
    text = "".join(parts).strip()
    if not text:
        raise SummaryRequestError("OpenAI response did not contain output text.")
    return text


def _request_openai(prompt: str, config: dict[str, Any]) -> str:
    keys = config.get("openai_api_keys") or []
    if not keys:
        raise SummaryConfigError(
            f"openai_api_key is required. Set it in {config.get('config_local_path')} or OPENAI_API_KEY."
        )
    payload: dict[str, Any] = {
        "model": config.get("openai_model", "gpt-5-mini"),
        "input": prompt,
    }
    if str(config.get("openai_instructions", "")).strip():
        payload["instructions"] = str(config["openai_instructions"]).strip()

    failures: list[str] = []
    for index, key in enumerate(keys, start=1):
        try:
            response = requests.post(
                f"{str(config.get('openai_base_url', 'https://api.openai.com/v1')).rstrip('/')}/responses",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=int(config.get("openai_timeout", 60)),
            )
            response.raise_for_status()
            return _extract_openai_text(response.json())
        except requests.RequestException as exc:
            failures.append(f"key#{index}: {exc.__class__.__name__}")
    raise SummaryRequestError("No OpenAI API key succeeded: " + "; ".join(failures))


def _request_deepseek(prompt: str, config: dict[str, Any]) -> str:
    keys = config.get("deepseek_api_keys") or []
    if not keys:
        raise SummaryConfigError(
            f"deepseek_api_key is required. Set it in {config.get('config_local_path')} or DEEPSEEK_API_KEY."
        )

    messages: list[dict[str, str]] = []
    if str(config.get("deepseek_instructions", "")).strip():
        messages.append({"role": "system", "content": str(config["deepseek_instructions"]).strip()})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": config.get("deepseek_model", "deepseek-v4-pro"),
        "messages": messages,
    }
    if config.get("deepseek_reasoning_effort"):
        payload["reasoning_effort"] = config["deepseek_reasoning_effort"]
    if config.get("deepseek_thinking_enabled", True):
        payload["thinking"] = {"type": "enabled"}

    failures: list[str] = []
    for index, key in enumerate(keys, start=1):
        try:
            response = requests.post(
                f"{str(config.get('deepseek_base_url', 'https://api.deepseek.com')).rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=int(config.get("deepseek_timeout", 60)),
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if text.strip():
                return text.strip()
            raise SummaryRequestError("DeepSeek response did not contain assistant text.")
        except requests.RequestException as exc:
            failures.append(f"key#{index}: {exc.__class__.__name__}")
    raise SummaryRequestError("No DeepSeek API key succeeded: " + "; ".join(failures))


def build_summary_prompt(
    config: EstimatorConfig,
    result: FinalResult,
    language: Language = "en",
) -> str:
    compact_result = result.model_dump(mode="json")
    compact_result["memory"].pop("raw_output", None)
    if compact_result.get("time"):
        compact_result["time"].pop("raw_output", None)

    language_instruction = (
        "Write the full answer in Simplified Chinese."
        if language == "zh"
        else "Write the full answer in English."
    )

    return f"""
You are analyzing an LLM training estimate for an engineer.

Return Markdown only. Keep it concise and practical.
{language_instruction}

Cover:
- whether the run is actionable,
- memory pressure and OOM risk,
- time and GPU-hour interpretation,
- cost interpretation,
- unsupported or approximate parts,
- concrete next steps to improve confidence or reduce risk.

Configuration:
{json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2)}

Estimator result:
{json.dumps(compact_result, ensure_ascii=False, indent=2)}
""".strip()


def generate_summary(
    config: EstimatorConfig,
    result: FinalResult,
    provider: Provider = "auto",
    language: Language = "en",
) -> tuple[str, str]:
    llm_config = load_reference_llm_config()
    resolved = _resolve_provider(llm_config, provider)
    prompt = build_summary_prompt(config, result, language)
    if resolved == "deepseek":
        return _request_deepseek(prompt, llm_config), resolved
    return _request_openai(prompt, llm_config), resolved

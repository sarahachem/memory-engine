from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from memory_engine.llm import LLMClient, LLMUsage, summarize_usage


def load_pricing(
    path: Path | None,
) -> dict[str, tuple[float, float, float]]:
    if path is None:
        return {}
    source = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, tuple[float, float, float]] = {}
    for model, rates in source.items():
        values = (
            float(rates["uncached_input"]),
            float(rates["cached_input"]),
            float(rates["output"]),
        )
        if any(value < 0 for value in values):
            raise ValueError("Pricing rates must not be negative.")
        result[model] = values
    return result


def estimate_cost_usd(
    usage: dict[str, Any],
    *,
    input_cost_per_million: float,
    cached_input_cost_per_million: float,
    output_cost_per_million: float,
) -> dict[str, Any]:
    """Apply explicit rates without coupling application code to pricing."""
    cached_tokens = int(usage.get("cached_input_tokens", 0))
    input_tokens = int(usage.get("input_tokens", 0))
    uncached_tokens = max(0, input_tokens - cached_tokens)
    output_tokens = int(usage.get("output_tokens", 0))
    estimated = (
        uncached_tokens * input_cost_per_million
        + cached_tokens * cached_input_cost_per_million
        + output_tokens * output_cost_per_million
    ) / 1_000_000
    return {
        "currency": "USD",
        "estimated_cost": round(estimated, 8),
        "rates_per_million_tokens": {
            "uncached_input": input_cost_per_million,
            "cached_input": cached_input_cost_per_million,
            "output": output_cost_per_million,
        },
    }


def attach_usage_and_rewrite(
    report: dict[str, Any],
    report_path: Path,
    llm_client: LLMClient,
    *,
    cost_rates: tuple[float, float, float] | None = None,
) -> None:
    drain = getattr(llm_client, "drain_usage", None)
    if not callable(drain):
        report["usage"] = {"available": False}
    else:
        report["usage"] = {
            "available": True,
            **summarize_usage(drain()),
        }
        model = getattr(llm_client, "model", None)
        if isinstance(model, str):
            report["usage"]["model"] = model
        if cost_rates is not None:
            report["usage"]["cost_estimate"] = estimate_cost_usd(
                report["usage"],
                input_cost_per_million=cost_rates[0],
                cached_input_cost_per_million=cost_rates[1],
                output_cost_per_million=cost_rates[2],
            )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def attach_named_usage_and_rewrite(
    report: dict[str, Any],
    report_path: Path,
    clients: dict[str, LLMClient],
    *,
    pricing_by_model: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Record both per-component and aggregate usage without double-draining."""
    usage_by_component: dict[str, dict[str, Any]] = {}
    all_records: list[LLMUsage] = []
    drained_by_identity: dict[int, tuple[LLMUsage, ...] | None] = {}
    aggregated_identities: set[int] = set()
    aggregate_costs: list[float] = []
    usage_available = False

    for name, client in clients.items():
        identity = id(client)
        if identity not in drained_by_identity:
            drain = getattr(client, "drain_usage", None)
            drained_by_identity[identity] = (
                tuple(drain()) if callable(drain) else None
            )
        records = drained_by_identity[identity]
        if records is None:
            usage_by_component[name] = {"available": False}
            continue
        usage_available = True
        usage_by_component[name] = {
            "available": True,
            **summarize_usage(records),
        }
        model = getattr(client, "model", None)
        if isinstance(model, str):
            usage_by_component[name]["model"] = model
            rates = (pricing_by_model or {}).get(model)
            if rates is not None:
                usage_by_component[name]["cost_estimate"] = estimate_cost_usd(
                    usage_by_component[name],
                    input_cost_per_million=rates[0],
                    cached_input_cost_per_million=rates[1],
                    output_cost_per_million=rates[2],
                )
        # A shared client belongs to the aggregate only once.
        if identity not in aggregated_identities:
            all_records.extend(records)
            if "cost_estimate" in usage_by_component[name]:
                aggregate_costs.append(
                    usage_by_component[name]["cost_estimate"][
                        "estimated_cost"
                    ]
                )
            aggregated_identities.add(identity)

    report["usage_by_component"] = usage_by_component
    report["usage"] = {
        "available": usage_available,
        **summarize_usage(tuple(all_records)),
    }
    if aggregate_costs:
        report["usage"]["estimated_cost_usd"] = round(
            sum(aggregate_costs), 8
        )
    else:
        report["usage"]["cost_estimate"] = {
            "available": False,
            "reason": "No explicit pricing rates were supplied.",
        }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal


def event_contract_fee(
    price: float,
    count: float,
    *,
    maker: bool = False,
    multiplier: float | None = None,
) -> float:
    """Return Kalshi's quadratic event-contract fee, rounded to a centicent.

    July 2026 general schedule: taker rate .07; maker rate .0175. The default
    general maker multiplier is zero, while taker multiplier is one. Callers
    should pass the series-specific multiplier when Kalshi publishes an override.
    """
    if not 0.0 <= price <= 1.0:
        raise ValueError("price must be between 0 and 1")
    if count < 0:
        raise ValueError("count cannot be negative")
    rate = Decimal("0.0175" if maker else "0.07")
    m = Decimal(str((0.0 if maker else 1.0) if multiplier is None else multiplier))
    raw = m * rate * Decimal(str(count)) * Decimal(str(price)) * (Decimal(1) - Decimal(str(price)))
    return float(raw.quantize(Decimal("0.0001"), rounding=ROUND_CEILING))


def net_edge(probability: float, price: float, count: int = 1, *, maker: bool = False) -> float:
    fee_per_contract = event_contract_fee(price, count, maker=maker) / max(count, 1)
    return probability - price - fee_per_contract

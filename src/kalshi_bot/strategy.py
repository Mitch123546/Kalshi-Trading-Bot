from __future__ import annotations

import math
from dataclasses import dataclass

from .fees import event_contract_fee
from .types import MarketView, Signal


@dataclass(frozen=True)
class RiskState:
    open_risk: float
    realized_pnl_today: float
    already_traded_market: bool


@dataclass(frozen=True)
class RiskLimits:
    max_order_risk: float
    max_daily_loss: float
    max_open_risk: float
    min_edge: float
    min_seconds_to_close: int
    max_seconds_to_close: int
    max_spread: float = 0.10
    max_price: float = 0.88
    min_price: float = 0.12


def propose_trade(
    view: MarketView,
    probability_yes: float,
    seconds_to_close: float,
    limits: RiskLimits,
    state: RiskState,
) -> Signal | None:
    if state.already_traded_market:
        return None
    if state.realized_pnl_today <= -limits.max_daily_loss:
        return None
    if state.open_risk >= limits.max_open_risk:
        return None
    if not limits.min_seconds_to_close <= seconds_to_close <= limits.max_seconds_to_close:
        return None
    if view.book.spread > limits.max_spread:
        return None

    yes_price = view.book.yes_ask
    no_price = 1.0 - view.book.yes_bid
    candidates = (("yes", probability_yes, yes_price), ("no", 1 - probability_yes, no_price))
    best = None
    for side, probability, price in candidates:
        if not limits.min_price <= price <= limits.max_price:
            continue
        fee = event_contract_fee(price, 1)
        edge = probability - price - fee
        if best is None or edge > best[3]:
            best = (side, probability, price, edge)
    if best is None or best[3] < limits.min_edge:
        return None

    side, probability, price, edge = best
    available_risk = min(limits.max_order_risk, limits.max_open_risk - state.open_risk)
    per_contract = price + event_contract_fee(price, 1)
    count = max(0, math.floor(available_risk / max(per_contract, 0.01)))
    if count < 1:
        return None
    return Signal(
        side=side,  # type: ignore[arg-type]
        probability=probability,
        price=price,
        edge=edge,
        count=count,
        reason=f"net edge {edge:.3f} after estimated taker fee",
    )

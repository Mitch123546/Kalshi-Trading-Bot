from __future__ import annotations

import math
from collections import deque
from datetime import UTC, datetime
from statistics import pstdev

from .types import MarketView

FEATURE_NAMES = (
    "distance_bps",
    "normalized_distance",
    "seconds_to_close_log",
    "return_5s",
    "return_15s",
    "return_30s",
    "return_60s",
    "return_180s",
    "rv_30s",
    "rv_60s",
    "rv_300s",
    "market_mid",
    "market_spread",
    "book_imbalance",
    "diffusion_probability",
    "market_minus_diffusion",
    "log_volume_24h",
    "log_open_interest",
    "hour_sin",
    "hour_cos",
)


class PriceHistory:
    def __init__(self, max_points: int = 1200):
        self.points: deque[tuple[float, float]] = deque(maxlen=max_points)

    def add(self, timestamp: datetime, price: float) -> None:
        ts = timestamp.timestamp()
        if self.points and ts < self.points[-1][0]:
            return
        if self.points and ts == self.points[-1][0]:
            self.points[-1] = (ts, price)
        else:
            self.points.append((ts, price))

    def price_ago(self, seconds: float) -> float | None:
        if not self.points:
            return None
        target = self.points[-1][0] - seconds
        return min(self.points, key=lambda pair: abs(pair[0] - target))[1]

    def log_return(self, seconds: float) -> float:
        old = self.price_ago(seconds)
        new = self.points[-1][1] if self.points else None
        if old is None or new is None or old <= 0 or new <= 0:
            return 0.0
        return math.log(new / old)

    def realized_vol(self, seconds: float) -> float:
        if len(self.points) < 3:
            return 0.0
        cutoff = self.points[-1][0] - seconds
        prices = [price for ts, price in self.points if ts >= cutoff]
        if len(prices) < 3:
            return 0.0
        returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        return pstdev(returns) if len(returns) >= 2 else 0.0


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def diffusion_probability(view: MarketView, history: PriceHistory, now: datetime) -> float:
    """No-drift diffusion baseline. It is a benchmark, not a claim of edge."""
    seconds = max(1.0, view.seconds_to_close(now))
    per_second_vol = max(history.realized_vol(300), 1e-8)
    sigma = per_second_vol * math.sqrt(seconds)
    distance = math.log(max(view.underlying, 1e-12) / max(view.target, 1e-12))
    return min(0.999, max(0.001, _norm_cdf(distance / sigma)))


def build_features(
    view: MarketView, history: PriceHistory, now: datetime | None = None
) -> dict[str, float]:
    now = (now or datetime.now(UTC)).astimezone(UTC)
    history.add(view.underlying_ts, view.underlying)
    seconds = max(1.0, view.seconds_to_close(now))
    rv60 = history.realized_vol(60)
    distance = math.log(max(view.underlying, 1e-12) / max(view.target, 1e-12))
    fair = diffusion_probability(view, history, now)
    hour = now.hour + now.minute / 60
    features = {
        "distance_bps": distance * 10_000,
        "normalized_distance": distance / max(rv60 * math.sqrt(seconds), 1e-6),
        "seconds_to_close_log": math.log1p(seconds),
        "return_5s": history.log_return(5),
        "return_15s": history.log_return(15),
        "return_30s": history.log_return(30),
        "return_60s": history.log_return(60),
        "return_180s": history.log_return(180),
        "rv_30s": history.realized_vol(30),
        "rv_60s": rv60,
        "rv_300s": history.realized_vol(300),
        "market_mid": view.book.mid,
        "market_spread": view.book.spread,
        "book_imbalance": view.book.imbalance,
        "diffusion_probability": fair,
        "market_minus_diffusion": view.book.mid - fair,
        "log_volume_24h": math.log1p(view.volume_24h),
        "log_open_interest": math.log1p(view.open_interest),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}

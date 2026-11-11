from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Side = Literal["yes", "no"]


@dataclass(frozen=True)
class Book:
    yes_bid: float
    yes_ask: float
    yes_bid_size: float
    yes_ask_size: float

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def spread(self) -> float:
        return max(0.0, self.yes_ask - self.yes_bid)

    @property
    def imbalance(self) -> float:
        total = self.yes_bid_size + self.yes_ask_size
        return 0.0 if total <= 0 else (self.yes_bid_size - self.yes_ask_size) / total


@dataclass(frozen=True)
class MarketView:
    series_ticker: str
    ticker: str
    event_ticker: str
    title: str
    open_time: datetime
    close_time: datetime
    target: float
    underlying: float
    underlying_ts: datetime
    volume_24h: float
    open_interest: float
    book: Book

    def seconds_to_close(self, now: datetime) -> float:
        return (self.close_time - now).total_seconds()


@dataclass(frozen=True)
class Prediction:
    probability_yes: float
    fair_probability_yes: float
    model_name: str
    features: dict[str, float]


@dataclass(frozen=True)
class Signal:
    side: Side
    probability: float
    price: float
    edge: float
    count: int
    reason: str

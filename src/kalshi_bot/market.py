from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from .types import Book, MarketView

COMMODITY_15M_SERIES = (
    "KXGOLD15M",
    "KXSILVER15M",
    "KXWTI15M",
    "KXBRENT15M",
    "KXCOPPER15M",
    "KXNATGAS15M",
    "KXPLATINUM15M",
    "KXPALLADIUM15M",
)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _number(data: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = data.get(name)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return default


def market_target(market: dict[str, Any]) -> float:
    for name in (
        "functional_strike",
        "floor_strike",
        "cap_strike",
        "strike_value",
        "target",
        "target_price",
    ):
        value = market.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    custom = market.get("custom_strike")
    if isinstance(custom, dict):
        value = custom.get("value") or custom.get("target")
        if value is not None:
            return float(value)
    raise ValueError(f"No numeric target found for {market.get('ticker')}")


def parse_orderbook(orderbook: dict[str, Any], market: dict[str, Any]) -> Book:
    """Normalize Kalshi's bid-only binary book into a YES bid/ask view."""

    def levels(side: str) -> list[tuple[float, float]]:
        raw = orderbook.get(f"{side}_dollars") or orderbook.get(side) or []
        parsed: list[tuple[float, float]] = []
        for level in raw:
            if isinstance(level, dict):
                price = _number(level, "price_dollars", "price", "0")
                size = _number(level, "count_fp", "quantity", "size", "1")
            else:
                price, size = float(level[0]), float(level[1])
                if price > 1:
                    price /= 100
            parsed.append((price, size))
        return parsed

    yes = levels("yes")
    no = levels("no")
    yes_bid, yes_bid_size = max(yes, default=(_number(market, "yes_bid_dollars"), 0.0))
    no_bid, no_bid_size = max(no, default=(_number(market, "no_bid_dollars"), 0.0))
    yes_ask = 1.0 - no_bid if no_bid > 0 else _number(market, "yes_ask_dollars", default=1.0)
    yes_ask_size = no_bid_size or _number(market, "yes_ask_size_fp")
    return Book(
        yes_bid=max(0.0, min(1.0, yes_bid)),
        yes_ask=max(0.0, min(1.0, yes_ask)),
        yes_bid_size=max(0.0, yes_bid_size),
        yes_ask_size=max(0.0, yes_ask_size),
    )


def extract_live_points(live_data: dict[str, Any]) -> list[tuple[datetime, float]]:
    """Extract timestamps/prices from Kalshi's intentionally schema-flexible live data."""
    details = live_data.get("details", live_data)
    candidates: list[Any] = []
    if isinstance(details, list):
        candidates = details
    elif isinstance(details, dict):
        for key in ("timeseries", "series", "data", "points", "prices", "index_values", "values"):
            value = details.get(key)
            if isinstance(value, list):
                candidates = value
                break
        if not candidates:
            # Some schemas nest the actual series one level deeper.
            for value in details.values():
                if isinstance(value, dict):
                    for key in ("timeseries", "data", "points", "prices", "values"):
                        nested = value.get(key)
                        if isinstance(nested, list):
                            candidates = nested
                            break
                if candidates:
                    break

    points: list[tuple[datetime, float]] = []
    for item in candidates:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            ts_raw, price_raw = item[0], item[1]
        elif isinstance(item, dict):
            timestamp_keys = (
                "t",
                "ts",
                "timestamp",
                "time",
                "observed_at",
                "created_time",
            )
            ts_raw = next(
                (item.get(key) for key in timestamp_keys if item.get(key) is not None),
                None,
            )
            price_keys = ("v", "value", "price", "index", "close", "mid")
            price_raw = next(
                (item.get(key) for key in price_keys if item.get(key) is not None),
                None,
            )
        else:
            continue
        try:
            if isinstance(ts_raw, (int, float)):
                seconds = float(ts_raw) / (1000 if float(ts_raw) > 10_000_000_000 else 1)
                ts = datetime.fromtimestamp(seconds, UTC)
            else:
                ts = parse_time(str(ts_raw))
            price = float(price_raw)
            if math.isfinite(price) and price > 0:
                points.append((ts, price))
        except (TypeError, ValueError, OSError):
            continue
    return sorted(set(points), key=lambda pair: pair[0])


def build_market_view(
    series_ticker: str,
    market: dict[str, Any],
    orderbook: dict[str, Any],
    live_data: dict[str, Any],
) -> MarketView:
    points = extract_live_points(live_data)
    if not points:
        raise ValueError(f"No underlying price points for {market.get('event_ticker')}")
    underlying_ts, underlying = points[-1]
    return MarketView(
        series_ticker=series_ticker,
        ticker=str(market["ticker"]),
        event_ticker=str(market["event_ticker"]),
        title=str(market.get("title") or market.get("subtitle") or market["ticker"]),
        open_time=parse_time(str(market["open_time"])),
        close_time=parse_time(str(market["close_time"])),
        target=market_target(market),
        underlying=underlying,
        underlying_ts=underlying_ts,
        volume_24h=_number(market, "volume_24h_fp", "volume_24h", "volume_fp", "volume"),
        open_interest=_number(market, "open_interest_fp", "open_interest"),
        book=parse_orderbook(orderbook, market),
    )


def liquidity_score(view: MarketView) -> float:
    """Operational score: rewards liquidity/depth and penalizes wide spreads."""
    spread_penalty = 80.0 * view.book.spread
    depth = view.book.yes_bid_size + view.book.yes_ask_size
    return math.log1p(view.volume_24h) + 0.25 * math.log1p(depth) - spread_penalty


def discover_15m_series(series: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [s for s in series if str(s.get("frequency", "")).lower() == "fifteen_min"],
        key=lambda s: str(s.get("ticker", "")),
    )

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kalshi_bot.config import Settings
from kalshi_bot.features import PriceHistory, build_features
from kalshi_bot.fees import event_contract_fee
from kalshi_bot.market import extract_live_points, market_target, parse_orderbook
from kalshi_bot.strategy import RiskLimits, RiskState, propose_trade
from kalshi_bot.types import Book, MarketView


def view(book: Book | None = None) -> MarketView:
    now = datetime.now(UTC)
    return MarketView(
        series_ticker="KXGOLD15M",
        ticker="GOLD-TEST",
        event_ticker="GOLD-EVENT",
        title="Gold test",
        open_time=now - timedelta(minutes=5),
        close_time=now + timedelta(minutes=5),
        target=2000.0,
        underlying=2001.0,
        underlying_ts=now,
        volume_24h=10000,
        open_interest=1000,
        book=book or Book(0.54, 0.56, 100, 120),
    )


def test_fee_matches_official_formula() -> None:
    assert event_contract_fee(0.50, 100) == 1.75
    assert event_contract_fee(0.10, 100) == 0.63
    assert event_contract_fee(0.50, 100, maker=True) == 0.0
    assert event_contract_fee(0.50, 100, maker=True, multiplier=1) == 0.4375


def test_orderbook_bid_only_conversion() -> None:
    market = {"yes_bid_dollars": "0.50", "yes_ask_dollars": "0.55"}
    book = parse_orderbook({"yes_dollars": [["0.52", "10"]], "no_dollars": [["0.45", "7"]]}, market)
    assert book.yes_bid == 0.52
    assert book.yes_ask == 0.55
    assert book.yes_ask_size == 7


def test_live_data_flexible_parser() -> None:
    points = extract_live_points(
        {"details": {"timeseries": [{"timestamp": "2026-01-01T00:00:00Z", "value": 2000.5}]}}
    )
    assert points[0][1] == 2000.5
    compact = extract_live_points(
        {"details": {"timeseries": [{"t": 1_767_225_600_000, "v": 2001.5}]}}
    )
    assert compact[0][1] == 2001.5


def test_market_target_variants() -> None:
    assert market_target({"functional_strike": "123.45"}) == 123.45
    assert market_target({"custom_strike": {"value": "12"}}) == 12.0


def test_strategy_uses_net_edge_and_risk_cap() -> None:
    signal = propose_trade(
        view(),
        probability_yes=0.75,
        seconds_to_close=300,
        limits=RiskLimits(2, 10, 10, 0.08, 45, 480),
        state=RiskState(0, 0, False),
    )
    assert signal is not None
    assert signal.side == "yes"
    assert signal.count <= 3


def test_live_requires_three_gates() -> None:
    settings = Settings(environment="prod", mode="live", allow_live=True)
    with pytest.raises(RuntimeError):
        settings.validate(confirm_live=False)


def test_feature_schema_is_finite() -> None:
    history = PriceHistory()
    item = view()
    for seconds in range(300, 0, -5):
        history.add(item.underlying_ts - timedelta(seconds=seconds), 2000 + seconds / 1000)
    features = build_features(item, history)
    assert all(value == value for value in features.values())

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict

from .api import KalshiClient
from .backtest import replay
from .config import PRODUCTION_REST, Settings, load_env_file
from .features import PriceHistory, build_features
from .market import build_market_view, discover_15m_series, extract_live_points
from .runner import run_bot
from .storage import Store
from .strategy import RiskLimits
from .train import train_model


async def research_scan() -> None:
    async with KalshiClient(PRODUCTION_REST) as client:
        series = discover_15m_series(await client.list_series("Commodities"))
        output = []
        for item in series:
            markets = await client.get_open_markets(item["ticker"])
            market = min(markets, key=lambda m: m.get("close_time", "9999")) if markets else None
            output.append(
                {
                    "ticker": item.get("ticker"),
                    "title": item.get("title"),
                    "settlement_sources": item.get("settlement_sources"),
                    "open_market": {
                        key: market.get(key)
                        for key in (
                            "ticker",
                            "event_ticker",
                            "close_time",
                            "yes_bid_dollars",
                            "yes_ask_dollars",
                            "volume_24h_fp",
                            "volume_fp",
                            "open_interest_fp",
                            "functional_strike",
                            "floor_strike",
                            "custom_strike",
                        )
                    }
                    if market
                    else None,
                }
            )
        print(json.dumps(output, indent=2, default=str))


async def settle(settings: Settings) -> None:
    store = Store(settings.db_path)
    labeled = 0
    try:
        async with KalshiClient(PRODUCTION_REST) as client:
            for ticker in store.unlabeled_closed_markets():
                market = await client.get_market(ticker)
                result = str(market.get("result", "")).lower()
                if result not in {"yes", "no"}:
                    continue
                store.label_market(ticker, int(result == "yes"))
                labeled += 1
    finally:
        store.close()
    print(json.dumps({"labeled_markets": labeled}))


async def snapshot(series: str) -> None:
    async with KalshiClient(PRODUCTION_REST) as client:
        markets = await client.get_open_markets(series)
        if not markets:
            raise RuntimeError(f"No open market for {series}")
        market = min(markets, key=lambda item: item.get("close_time", "9999"))
        orderbook, live_data = await asyncio.gather(
            client.get_orderbook(market["ticker"]),
            client.get_event_live_data(market["event_ticker"]),
        )
        view = build_market_view(series, market, orderbook, live_data)
        history = PriceHistory()
        for timestamp, price in extract_live_points(live_data):
            history.add(timestamp, price)
        print(
            json.dumps(
                {
                    "view": asdict(view),
                    "features": build_features(view, history),
                    "live_data_type": live_data.get("type"),
                    "live_points": len(extract_live_points(live_data)),
                },
                indent=2,
                default=str,
            )
        )


def limits(settings: Settings) -> RiskLimits:
    return RiskLimits(
        max_order_risk=settings.max_order_risk,
        max_daily_loss=settings.max_daily_loss,
        max_open_risk=settings.max_open_risk,
        min_edge=settings.min_edge,
        min_seconds_to_close=settings.min_seconds_to_close,
        max_seconds_to_close=settings.max_seconds_to_close,
    )


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(
        prog="kalshi-bot", description="Research-first Kalshi 15-minute commodity bot"
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan", help="List current 15-minute commodity series from production")
    snapshot_parser = sub.add_parser("snapshot", help="Normalize one current live market")
    snapshot_parser.add_argument("--series", default="KXWTI15M")
    run_parser = sub.add_parser("run", help="Collect data and optionally paper/live trade")
    run_parser.add_argument("--series", default="auto", help="Kalshi series ticker or auto")
    run_parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required final acknowledgement when KALSHI_MODE=live",
    )
    sub.add_parser("settle", help="Label stored snapshots using settled Kalshi results")
    train_parser = sub.add_parser("train", help="Train/calibrate the local PyTorch model")
    train_parser.add_argument("--epochs", type=int, default=150)
    train_parser.add_argument("--hidden", type=int, default=96)
    sub.add_parser("backtest", help="Replay labeled snapshots with executable prices and fees")
    sub.add_parser("metrics", help="Rank series by out-of-sample Brier score after 30 markets")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()

    if args.command == "scan":
        asyncio.run(research_scan())
        return
    if args.command == "snapshot":
        asyncio.run(snapshot(args.series))
        return
    if args.command == "run":
        settings.validate(confirm_live=args.confirm_live)
        asyncio.run(run_bot(settings, series=args.series))
        return
    settings.validate()
    if args.command == "settle":
        asyncio.run(settle(settings))
        return
    store = Store(settings.db_path)
    try:
        if args.command == "train":
            report = train_model(store, settings.model_path, epochs=args.epochs, hidden=args.hidden)
            print(json.dumps(asdict(report), indent=2))
        elif args.command == "backtest":
            print(json.dumps(asdict(replay(store, limits(settings))), indent=2))
        elif args.command == "metrics":
            print(json.dumps(store.series_metrics(30), indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import logging
import signal
import uuid
from datetime import UTC, datetime

from .api import KalshiAPIError, KalshiClient
from .config import Settings
from .features import PriceHistory, build_features
from .fees import event_contract_fee
from .market import (
    COMMODITY_15M_SERIES,
    build_market_view,
    discover_15m_series,
    extract_live_points,
)
from .model import load_model
from .storage import Store
from .strategy import RiskLimits, RiskState, propose_trade
from .types import Prediction

log = logging.getLogger(__name__)


class Bot:
    def __init__(self, settings: Settings, *, series: str = "KXGOLD15M"):
        self.settings = settings
        self.series = series
        self.store = Store(settings.db_path)
        self.model = load_model(settings.model_path)
        self.histories: dict[str, PriceHistory] = {}
        self.stop = asyncio.Event()
        self.selected_series = "KXGOLD15M" if series == "auto" else series
        self.last_selection = 0.0
        self.last_catalog_refresh = 0.0
        self.available_series = list(COMMODITY_15M_SERIES)

    async def close(self) -> None:
        self.store.close()

    async def _select_series(self, client: KalshiClient) -> str:
        if self.series != "auto":
            return self.series
        now = asyncio.get_running_loop().time()
        if now - self.last_selection < 60:
            return self.selected_series
        if now - self.last_catalog_refresh >= 3600:
            try:
                catalog = discover_15m_series(await client.list_series("Commodities"))
                discovered = [str(item["ticker"]) for item in catalog]
                if discovered:
                    self.available_series = discovered
                    self.last_catalog_refresh = now
            except KalshiAPIError as exc:
                log.warning("Catalog refresh failed; using cached series: %s", exc)
        results = await asyncio.gather(
            *(client.get_open_markets(s) for s in self.available_series),
            return_exceptions=True,
        )
        scored: list[tuple[float, str]] = []
        for series, result in zip(self.available_series, results, strict=True):
            if isinstance(result, Exception) or not result:
                continue
            market = min(result, key=lambda m: m.get("close_time", "9999"))
            bid = float(market.get("yes_bid_dollars") or 0)
            ask = float(market.get("yes_ask_dollars") or 1)
            volume = float(market.get("volume_24h_fp") or market.get("volume_fp") or 0)
            score = __import__("math").log1p(volume) - 80 * max(0, ask - bid)
            # Once enough local outcomes exist, prefer better calibrated series.
            metric = next(
                (m for m in self.store.series_metrics(30) if m["series_ticker"] == series), None
            )
            if metric:
                score += 2.0 * (0.25 - float(metric["brier"]))
            scored.append((score, series))
        if scored:
            self.selected_series = max(scored)[1]
            self.last_selection = now
        return self.selected_series

    async def _current_view(self, client: KalshiClient, series: str):
        markets = await client.get_open_markets(series)
        now = datetime.now(UTC)
        candidates = []
        for market in markets:
            try:
                close = datetime.fromisoformat(str(market["close_time"]).replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            if close > now:
                candidates.append((close, market))
        if not candidates:
            return None
        market = min(candidates, key=lambda pair: pair[0])[1]
        orderbook, live_data = await asyncio.gather(
            client.get_orderbook(market["ticker"]),
            client.get_event_live_data(market["event_ticker"]),
        )
        history = self.histories.setdefault(series, PriceHistory())
        for timestamp, price in extract_live_points(live_data):
            history.add(timestamp, price)
        return build_market_view(series, market, orderbook, live_data)

    def _limits(self) -> RiskLimits:
        s = self.settings
        return RiskLimits(
            max_order_risk=s.max_order_risk,
            max_daily_loss=s.max_daily_loss,
            max_open_risk=s.max_open_risk,
            min_edge=s.min_edge,
            min_seconds_to_close=s.min_seconds_to_close,
            max_seconds_to_close=s.max_seconds_to_close,
        )

    async def run(self) -> None:
        signer_args = {
            "key_id": self.settings.api_key_id,
            "private_key_path": self.settings.private_key_path,
        }
        async with KalshiClient(self.settings.rest_url, **signer_args) as client:
            while not self.stop.is_set():
                try:
                    series = await self._select_series(client)
                    view = await self._current_view(client, series)
                    if view is None:
                        log.info("No open market for %s", series)
                        await asyncio.sleep(max(5.0, self.settings.poll_seconds))
                        continue
                    history = self.histories.setdefault(series, PriceHistory())
                    features = build_features(view, history)
                    probability = self.model.predict(features)
                    prediction = Prediction(
                        probability_yes=probability,
                        fair_probability_yes=features["diffusion_probability"],
                        model_name=self.model.name,
                        features=features,
                    )
                    self.store.save_snapshot(view, prediction)
                    seconds = view.seconds_to_close(datetime.now(UTC))
                    signal_ = propose_trade(
                        view,
                        probability,
                        seconds,
                        self._limits(),
                        RiskState(
                            open_risk=self.store.open_risk(),
                            realized_pnl_today=self.store.realized_pnl_today(),
                            already_traded_market=self.store.has_traded_market(view.ticker),
                        ),
                    )
                    log.info(
                        "%s %s spot=%.5f target=%.5f p_yes=%.3f book=%.2f/%.2f t=%.0fs%s",
                        series,
                        view.ticker,
                        view.underlying,
                        view.target,
                        probability,
                        view.book.yes_bid,
                        view.book.yes_ask,
                        seconds,
                        f" signal={signal_.side} edge={signal_.edge:.3f}" if signal_ else "",
                    )
                    if signal_ and self.settings.mode == "paper":
                        fee = event_contract_fee(signal_.price, signal_.count)
                        self.store.open_paper_position(view, signal_, fee)
                        self.store.save_decision(
                            "paper", view.ticker, "paper_fill", signal_.reason, signal_
                        )
                    elif signal_ and self.settings.mode == "live":
                        # V2 quotes the YES book: buying NO is an ask on YES at 1-NO.
                        order_side = "bid" if signal_.side == "yes" else "ask"
                        order_price = signal_.price if signal_.side == "yes" else 1 - signal_.price
                        response = await client.create_order_v2(
                            ticker=view.ticker,
                            side=order_side,
                            count=signal_.count,
                            price=order_price,
                            client_order_id=str(uuid.uuid4()),
                        )
                        self.store.save_decision(
                            "live", view.ticker, "live_order", signal_.reason, signal_, response
                        )
                except (KalshiAPIError, ValueError, KeyError) as exc:
                    log.warning("Cycle skipped: %s", exc)
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=self.settings.poll_seconds)
                except TimeoutError:
                    pass


async def run_bot(settings: Settings, *, series: str) -> None:
    bot = Bot(settings, series=series)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.stop.set)
        except NotImplementedError:
            pass
    try:
        await bot.run()
    finally:
        await bot.close()

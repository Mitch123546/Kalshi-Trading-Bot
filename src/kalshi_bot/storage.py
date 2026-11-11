from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .model import feature_json
from .types import MarketView, Prediction, Signal

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY,
    observed_at TEXT NOT NULL,
    series_ticker TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    event_ticker TEXT NOT NULL,
    close_time TEXT NOT NULL,
    target REAL NOT NULL,
    underlying REAL NOT NULL,
    yes_bid REAL NOT NULL,
    yes_ask REAL NOT NULL,
    yes_bid_size REAL NOT NULL,
    yes_ask_size REAL NOT NULL,
    model_name TEXT NOT NULL,
    probability_yes REAL NOT NULL,
    fair_probability_yes REAL NOT NULL,
    features_json TEXT NOT NULL,
    outcome INTEGER,
    UNIQUE(market_ticker, observed_at)
);

CREATE INDEX IF NOT EXISTS snapshots_market_time
ON snapshots(market_ticker, observed_at);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    side TEXT,
    price REAL,
    probability REAL,
    edge REAL,
    count INTEGER,
    reason TEXT NOT NULL,
    external_order_id TEXT,
    raw_response_json TEXT
);

CREATE TABLE IF NOT EXISTS paper_positions (
    market_ticker TEXT PRIMARY KEY,
    series_ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    count INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_fee REAL NOT NULL,
    opened_at TEXT NOT NULL,
    outcome INTEGER,
    realized_pnl REAL
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def save_snapshot(
        self, view: MarketView, prediction: Prediction, observed_at: datetime | None = None
    ) -> None:
        now = (observed_at or datetime.now(UTC)).isoformat()
        self.db.execute(
            """
            INSERT OR IGNORE INTO snapshots (
                observed_at, series_ticker, market_ticker, event_ticker, close_time,
                target, underlying, yes_bid, yes_ask, yes_bid_size, yes_ask_size,
                model_name, probability_yes, fair_probability_yes, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                view.series_ticker,
                view.ticker,
                view.event_ticker,
                view.close_time.isoformat(),
                view.target,
                view.underlying,
                view.book.yes_bid,
                view.book.yes_ask,
                view.book.yes_bid_size,
                view.book.yes_ask_size,
                prediction.model_name,
                prediction.probability_yes,
                prediction.fair_probability_yes,
                feature_json(prediction.features),
            ),
        )
        self.db.commit()

    def save_decision(
        self,
        mode: str,
        market_ticker: str,
        action: str,
        reason: str,
        signal: Signal | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO decisions (
                created_at, mode, market_ticker, action, side, price, probability,
                edge, count, reason, external_order_id, raw_response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                mode,
                market_ticker,
                action,
                signal.side if signal else None,
                signal.price if signal else None,
                signal.probability if signal else None,
                signal.edge if signal else None,
                signal.count if signal else None,
                reason,
                (response or {}).get("order_id"),
                json.dumps(response, default=str) if response else None,
            ),
        )
        self.db.commit()

    def open_paper_position(self, view: MarketView, signal: Signal, fee: float) -> None:
        self.db.execute(
            """
            INSERT OR IGNORE INTO paper_positions (
                market_ticker, series_ticker, side, count, entry_price, entry_fee, opened_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view.ticker,
                view.series_ticker,
                signal.side,
                signal.count,
                signal.price,
                fee,
                datetime.now(UTC).isoformat(),
            ),
        )
        self.db.commit()

    def has_position(self, market_ticker: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM paper_positions WHERE market_ticker=? LIMIT 1", (market_ticker,)
        ).fetchone()
        return row is not None

    def has_traded_market(self, market_ticker: str) -> bool:
        if self.has_position(market_ticker):
            return True
        row = self.db.execute(
            "SELECT 1 FROM decisions WHERE market_ticker=? "
            "AND action IN ('paper_fill', 'live_order') LIMIT 1",
            (market_ticker,),
        ).fetchone()
        return row is not None

    def open_risk(self) -> float:
        row = self.db.execute(
            "SELECT COALESCE(SUM(entry_price * count + entry_fee), 0) AS risk "
            "FROM paper_positions WHERE outcome IS NULL"
        ).fetchone()
        return float(row["risk"])

    def realized_pnl_today(self) -> float:
        today = datetime.now(UTC).date().isoformat()
        row = self.db.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS pnl FROM paper_positions "
            "WHERE outcome IS NOT NULL AND substr(opened_at, 1, 10)=?",
            (today,),
        ).fetchone()
        return float(row["pnl"])

    def label_market(self, market_ticker: str, outcome: int) -> None:
        if outcome not in {0, 1}:
            raise ValueError("outcome must be 0 or 1")
        self.db.execute(
            "UPDATE snapshots SET outcome=? WHERE market_ticker=?", (outcome, market_ticker)
        )
        rows = self.db.execute(
            "SELECT * FROM paper_positions WHERE market_ticker=? AND outcome IS NULL",
            (market_ticker,),
        ).fetchall()
        for row in rows:
            won = (row["side"] == "yes" and outcome == 1) or (row["side"] == "no" and outcome == 0)
            payout = row["count"] if won else 0.0
            pnl = payout - row["entry_price"] * row["count"] - row["entry_fee"]
            self.db.execute(
                "UPDATE paper_positions SET outcome=?, realized_pnl=? WHERE market_ticker=?",
                (outcome, pnl, market_ticker),
            )
        self.db.commit()

    def unlabeled_closed_markets(self, now: datetime | None = None) -> list[str]:
        now_text = (now or datetime.now(UTC)).isoformat()
        rows = self.db.execute(
            "SELECT DISTINCT market_ticker FROM snapshots "
            "WHERE outcome IS NULL AND close_time < ? ORDER BY close_time",
            (now_text,),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def labeled_samples(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM snapshots WHERE outcome IS NOT NULL ORDER BY close_time, observed_at"
        ).fetchall()

    def series_metrics(self, minimum_markets: int = 30) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """
            SELECT series_ticker,
                   COUNT(DISTINCT market_ticker) AS markets,
                   COUNT(*) AS samples,
                   AVG((probability_yes - outcome) * (probability_yes - outcome)) AS brier
            FROM snapshots WHERE outcome IS NOT NULL
            GROUP BY series_ticker HAVING COUNT(DISTINCT market_ticker) >= ?
            ORDER BY brier ASC
            """,
            (minimum_markets,),
        ).fetchall()
        return [dict(row) for row in rows]

    def iter_market_snapshots(self) -> Iterable[list[sqlite3.Row]]:
        tickers = self.db.execute(
            "SELECT DISTINCT market_ticker FROM snapshots "
            "WHERE outcome IS NOT NULL ORDER BY close_time"
        ).fetchall()
        for ticker in tickers:
            yield self.db.execute(
                "SELECT * FROM snapshots WHERE market_ticker=? ORDER BY observed_at",
                (ticker[0],),
            ).fetchall()

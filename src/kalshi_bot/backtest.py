from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .fees import event_contract_fee
from .strategy import RiskLimits, RiskState, propose_trade
from .types import Book, MarketView


@dataclass(frozen=True)
class BacktestResult:
    trades: int
    wins: int
    pnl: float
    roi: float
    max_drawdown: float
    brier: float


def replay(store, limits: RiskLimits) -> BacktestResult:
    trades = wins = 0
    pnl = cost = peak = equity = max_drawdown = 0.0
    squared_errors: list[float] = []
    for rows in store.iter_market_snapshots():
        if not rows:
            continue
        outcome = int(rows[0]["outcome"])
        chosen = None
        for row in rows:
            probability = float(row["probability_yes"])
            squared_errors.append((probability - outcome) ** 2)
            close_time = datetime.fromisoformat(row["close_time"])
            observed_at = datetime.fromisoformat(row["observed_at"])
            view = MarketView(
                series_ticker=row["series_ticker"],
                ticker=row["market_ticker"],
                event_ticker=row["event_ticker"],
                title=row["market_ticker"],
                open_time=observed_at,
                close_time=close_time,
                target=float(row["target"]),
                underlying=float(row["underlying"]),
                underlying_ts=observed_at,
                volume_24h=0,
                open_interest=0,
                book=Book(
                    float(row["yes_bid"]),
                    float(row["yes_ask"]),
                    float(row["yes_bid_size"]),
                    float(row["yes_ask_size"]),
                ),
            )
            chosen = propose_trade(
                view,
                probability,
                (close_time - observed_at).total_seconds(),
                limits,
                RiskState(open_risk=0, realized_pnl_today=0, already_traded_market=False),
            )
            if chosen:
                break
        if not chosen:
            continue
        fee = event_contract_fee(chosen.price, chosen.count)
        trade_cost = chosen.price * chosen.count + fee
        won = (chosen.side == "yes" and outcome == 1) or (chosen.side == "no" and outcome == 0)
        trade_pnl = (chosen.count if won else 0) - trade_cost
        trades += 1
        wins += int(won)
        cost += trade_cost
        pnl += trade_pnl
        equity += trade_pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return BacktestResult(
        trades=trades,
        wins=wins,
        pnl=pnl,
        roi=pnl / cost if cost else 0.0,
        max_drawdown=max_drawdown,
        brier=sum(squared_errors) / len(squared_errors) if squared_errors else float("nan"),
    )

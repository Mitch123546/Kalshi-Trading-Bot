# Kalshi 15-minute commodity bot

A research-first Python system for collecting, modeling, paper trading, and (only after explicit opt-in) trading Kalshi's rolling 15-minute commodity event contracts.

**[Open the Kalshi Signal Lab website](https://mitch123546.github.io/Kalshi-Trading-Bot/)**

The default is **automatic market selection**. On the research snapshot taken November 11, 2026, Kalshi's category page showed gold with the most aggregate displayed commodity volume, but the live API showed that gold was between active sessions while WTI, Brent, and natural gas were open. `--series auto` therefore discovers the live catalog and periodically re-scores only available markets by volume, spread, and—after enough outcomes—local calibration.

> This is experimental software, not a promise of profit or financial advice. Fifteen-minute price direction is noisy, fees and adverse selection are material, and a high hit rate can still lose money when contracts are expensive.

## What the research found

- Kalshi currently offers 15-minute commodity markets for gold, silver, WTI, Brent, copper, natural gas, platinum, palladium, and others. The exact catalog is discovered from Kalshi's API rather than frozen in code. See Kalshi's [commodity market page](https://kalshi.com/category/commodities) and [15-minute market guide](https://kalshi.com/pro/help/fifteen-minute-markets).
- These commodity contracts use published price feeds—typically Pyth feeds—as their settlement source. The bot uses Kalshi's own event live-data endpoint so the feature price is aligned with the market's displayed underlying. The [`scan` command](#research-scan) prints each series' current source.
- The most credible public 15-minute experiment I found is candid that it did **not** make money after studying thousands of settled markets and millions of trades: [aaryann123/kalshi-bot](https://github.com/aaryann123/kalshi-bot). Other repositories contain useful engineering patterns but rarely provide auditable live results. Examples reviewed include [kapelame/kalshi-crypto-bot](https://github.com/kapelame/kalshi-crypto-bot) and [oribarlevco-cell/kalshi-btc-15m-bot](https://github.com/oribarlevco-cell/kalshi-btc-15m-bot).
- Kalshi's current general taker fee is `ceil_to_centicent(0.07 × contracts × price × (1-price))`; the maker formula uses `0.0175` and a series multiplier that is usually zero unless listed otherwise. The implementation models the fee exactly and uses executable bid/ask prices, not midpoint fantasy fills. See the [official fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf).
- Public market data requires no API key. Authenticated requests use asymmetric keys; Kalshi now recommends Ed25519. Orders use the current V2 event-order endpoint. See [market-data quick start](https://docs.kalshi.com/getting_started/quick_start_market_data), [API keys](https://docs.kalshi.com/getting_started/api_keys), and [Create Order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2).

The practical conclusion is to **collect first, select the market empirically, and refuse to trade without a fee-adjusted edge**. The local neural model is useful only if it beats both the market price and the diffusion baseline on untouched, chronologically later markets.

## Design

- Polls the current market, bid-only order book, and underlying event time series from Kalshi's production data API.
- Stores raw market snapshots, features, predictions, decisions, fills, and outcomes in SQLite/WAL.
- Builds price-distance, multi-horizon momentum, realized-volatility, time-to-expiry, spread, depth imbalance, volume, and time-of-day features.
- Includes a no-drift diffusion probability as a transparent baseline.
- Trains a compact calibrated PyTorch MLP using a chronological **market-level** train/validation/test split. A 3070 is more than sufficient; CUDA is selected automatically.
- Paper fills cross the displayed spread and include the official quadratic taker fee.
- Backtests allow only one trade per market and replay the quote that was actually observed.
- `auto` selection starts with liquidity and incorporates per-series Brier scores only after 30 settled local markets.
- Production is protected by three independent gates: `KALSHI_ENV=prod`, `KALSHI_ALLOW_LIVE=true`, and `--confirm-live`.

This is intentionally not an LLM making a fresh verbal guess every 15 minutes. A small numerical model trained on aligned microstructure data is faster, testable, and much harder to fool with data leakage.

## Setup

Python 3.11+ is required.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -e ".[ml,dev]"
copy .env.example .env
```

For public production collection and scanning, no credentials are needed. This is why the example defaults to `KALSHI_ENV=prod` even though order placement remains disabled. If you separately experiment with Kalshi's demo environment, demo and production credentials are distinct; the demo endpoint is documented [here](https://docs.kalshi.com/getting_started/demo_env).

## Research scan

```bash
kalshi-bot scan
kalshi-bot snapshot --series KXWTI15M
```

This queries the live production catalog and prints every commodity series whose frequency is `fifteen_min`, its settlement source, active ticker, spread, volume, open interest, and strike fields. Run it before changing the configured series—Kalshi can add, rename, pause, or remove products.

## Collect before training

Use the live selector, or pin a series for a controlled experiment:

```bash
# .env: KALSHI_MODE=collect
kalshi-bot run --series auto

# controlled single-series collection
kalshi-bot run --series KXGOLD15M
```

Stop with Ctrl+C. Then label markets after they settle:

```bash
kalshi-bot settle
kalshi-bot metrics
```

Collecting one series continuously for at least several market regimes is much more useful than immediately fitting a GPU model to a day of highly correlated snapshots. The trainer refuses to run with fewer than 30 settled markets; in practice, use hundreds or thousands.

## Train locally on the 3070

```bash
kalshi-bot train --epochs 150 --hidden 96
```

The model and a JSON test report are written under `models/`. Training splits chronologically by complete market, fits normalization on the training period only, tunes temperature on validation markets, and reports the untouched test Brier score. CUDA is used when available.

Do not deploy a model merely because accuracy exceeds 50%. Require all of the following:

1. Test Brier score better than the diffusion baseline and the market midpoint.
2. Positive fee-adjusted P&L under executable bid/ask replay.
3. A sufficiently narrow bootstrap confidence interval across complete markets.
4. Positive forward paper results after the model was frozen.

## Paper trade and backtest

```bash
# .env: KALSHI_MODE=paper
kalshi-bot run --series auto
kalshi-bot settle
kalshi-bot backtest
```

Important limitation: only snapshots collected by this program contain the aligned underlying, book, and model features needed for an honest replay. Kalshi's historical candle endpoints do not recreate queue position or every quote you could have traded.

## Live mode

Keep live mode off until forward paper evidence is convincing. When you intentionally enable it:

```bash
# .env
KALSHI_ENV=prod
KALSHI_MODE=live
KALSHI_ALLOW_LIVE=true
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PATH=C:\secure\kalshi.key

kalshi-bot run --series auto --confirm-live
```

Live orders are immediate-or-cancel limit orders; the bot places at most one order per market. Defaults cap each order's worst-case cost at $2, total open risk at $10, and the daily paper-loss stop at $10. Review and lower these values before first use. API errors, missing underlying data, stale/closed markets, wide spreads, and insufficient edge all fail closed.

## Project layout

```text
src/kalshi_bot/
  api.py         current REST authentication, market data, and V2 orders
  market.py      discovery and schema normalization
  features.py    causal feature calculation and diffusion benchmark
  model.py       baseline and calibrated PyTorch artifact loader
  train.py       chronological GPU/CPU training
  strategy.py    fee-adjusted signal and risk gates
  storage.py     SQLite event log and paper ledger
  backtest.py    executable-price replay
  runner.py      collection/paper/live loop
  cli.py         command-line interface
tests/           fee, parsing, feature, strategy, and safety tests
```

## Known limitations

- The first version polls once per second. That is appropriate for research and conservative entry but not a claim of low-latency market making. Kalshi WebSockets are the next optimization after a strategy survives paper testing.
- IOC fills can be partial. The exchange response is logged, but a production deployment should also reconcile the user-fill WebSocket and positions before increasing limits.
- Market selection is deliberately conservative and data-hungry. “Easiest to predict” is not known in advance; liquidity, calibration, and net P&L can rank differently.
- The bot does not use news or an LLM sentiment feed. At a 15-minute horizon, timestamp alignment and execution typically matter more, and external text can introduce latency and leakage.
- Rules, API shapes, fees, and series can change. Re-run `scan`, read the current contract rules, and verify the current fee schedule before live use.

## License

MIT. You are responsible for compliance with Kalshi's rules, applicable law, and your own risk limits.

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES
from .model import _build_torch_net


@dataclass(frozen=True)
class TrainReport:
    device: str
    markets_train: int
    markets_validation: int
    markets_test: int
    samples_train: int
    brier_validation: float
    brier_test: float
    accuracy_test: float
    temperature: float
    epochs: int


def _market_split(rows, seed: int = 7):
    market_order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        ticker = str(row["market_ticker"])
        if ticker not in seen:
            seen.add(ticker)
            market_order.append(ticker)
    if len(market_order) < 30:
        raise ValueError("Need at least 30 settled markets before training")
    n = len(market_order)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    groups = (
        set(market_order[:train_end]),
        set(market_order[train_end:val_end]),
        set(market_order[val_end:]),
    )
    # Limit autocorrelation and class domination: one row per 5-second bucket per market.
    output = []
    for group in groups:
        bucketed = {}
        for row in rows:
            if row["market_ticker"] not in group:
                continue
            timestamp = datetime.fromisoformat(str(row["observed_at"])).timestamp()
            bucket = int(timestamp // 5)
            bucketed[(row["market_ticker"], bucket)] = row
        output.append(list(bucketed.values()))
    return output, tuple(len(group) for group in groups)


def _arrays(rows):
    x = np.asarray(
        [[json.loads(row["features_json"])[name] for name in FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([row["outcome"] for row in rows], dtype=np.float32)
    return x, y


def train_model(
    store,
    output_path: Path,
    *,
    epochs: int = 150,
    hidden: int = 96,
    seed: int = 7,
) -> TrainReport:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Install the ML extras: pip install -e .[ml]") from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows = store.labeled_samples()
    (train_rows, val_rows, test_rows), market_counts = _market_split(rows, seed)
    x_train, y_train = _arrays(train_rows)
    x_val, y_val = _arrays(val_rows)
    x_test, y_test = _arrays(test_rows)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_train, x_val, x_test = ((x - mean) / scale for x in (x_train, x_val, x_test))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _build_torch_net(len(FEATURE_NAMES), hidden).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-3)
    train_x = torch.tensor(x_train, device=device)
    train_y = torch.tensor(y_train, device=device).unsqueeze(1)
    val_x = torch.tensor(x_val, device=device)
    val_y = torch.tensor(y_val, device=device).unsqueeze(1)
    best_loss = float("inf")
    best_state = None
    patience = 20
    stale = 0
    batch_size = min(2048, len(train_x))
    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        net.train()
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = functional.binary_cross_entropy_with_logits(net(train_x[idx]), train_y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            val_loss = functional.binary_cross_entropy_with_logits(net(val_x), val_y).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in net.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training failed to produce a model")
    net.load_state_dict(best_state)

    # Calibrate on validation markets only; a small grid is stable and dependency-free.
    net.eval()
    with torch.inference_mode():
        val_logits = net(val_x).squeeze().cpu().numpy()
        test_logits = net(torch.tensor(x_test, device=device)).squeeze().cpu().numpy()
    temperatures = np.geomspace(0.35, 3.0, 100)
    val_losses = []
    for temp in temperatures:
        probs = 1 / (1 + np.exp(-np.clip(val_logits / temp, -30, 30)))
        val_losses.append(np.mean((probs - y_val) ** 2))
    temperature = float(temperatures[int(np.argmin(val_losses))])
    val_probs = 1 / (1 + np.exp(-np.clip(val_logits / temperature, -30, 30)))
    test_probs = 1 / (1 + np.exp(-np.clip(test_logits / temperature, -30, 30)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature_names": FEATURE_NAMES,
            "hidden": hidden,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "temperature": temperature,
            "state_dict": best_state,
        },
        output_path,
    )
    report = TrainReport(
        device=device,
        markets_train=market_counts[0],
        markets_validation=market_counts[1],
        markets_test=market_counts[2],
        samples_train=len(train_rows),
        brier_validation=float(np.mean((val_probs - y_val) ** 2)),
        brier_test=float(np.mean((test_probs - y_test) ** 2)),
        accuracy_test=float(np.mean((test_probs >= 0.5) == y_test)),
        temperature=temperature,
        epochs=epochs_run,
    )
    output_path.with_suffix(".report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    return report

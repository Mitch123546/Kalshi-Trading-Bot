from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PRODUCTION_REST = "https://external-api.kalshi.com/trade-api/v2"
DEMO_REST = "https://external-api.demo.kalshi.co/trade-api/v2"


def load_env_file(path: Path = Path(".env")) -> None:
    """Load a simple KEY=VALUE file without overriding the process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    environment: str = "prod"
    mode: str = "paper"
    api_key_id: str | None = None
    private_key_path: Path | None = None
    allow_live: bool = False
    max_order_risk: float = 2.0
    max_daily_loss: float = 10.0
    max_open_risk: float = 10.0
    min_edge: float = 0.08
    min_seconds_to_close: int = 45
    max_seconds_to_close: int = 480
    db_path: Path = Path("data/kalshi.db")
    model_path: Path = Path("models/commodity_mlp.pt")
    poll_seconds: float = 1.0

    @property
    def rest_url(self) -> str:
        return PRODUCTION_REST if self.environment == "prod" else DEMO_REST

    @classmethod
    def from_env(cls) -> Settings:
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        return cls(
            environment=os.getenv("KALSHI_ENV", "prod").strip().lower(),
            mode=os.getenv("KALSHI_MODE", "paper").strip().lower(),
            api_key_id=os.getenv("KALSHI_API_KEY_ID") or None,
            private_key_path=Path(key_path) if key_path else None,
            allow_live=_bool("KALSHI_ALLOW_LIVE"),
            max_order_risk=float(os.getenv("KALSHI_MAX_ORDER_RISK_DOLLARS", "2.00")),
            max_daily_loss=float(os.getenv("KALSHI_MAX_DAILY_LOSS_DOLLARS", "10.00")),
            max_open_risk=float(os.getenv("KALSHI_MAX_OPEN_RISK_DOLLARS", "10.00")),
            min_edge=float(os.getenv("KALSHI_MIN_EDGE", "0.08")),
            min_seconds_to_close=int(os.getenv("KALSHI_MIN_SECONDS_TO_CLOSE", "45")),
            max_seconds_to_close=int(os.getenv("KALSHI_MAX_SECONDS_TO_CLOSE", "480")),
            db_path=Path(os.getenv("KALSHI_DB_PATH", "data/kalshi.db")),
            model_path=Path(os.getenv("KALSHI_MODEL_PATH", "models/commodity_mlp.pt")),
            poll_seconds=float(os.getenv("KALSHI_POLL_SECONDS", "1.0")),
        )

    def validate(self, *, confirm_live: bool = False) -> None:
        if self.environment not in {"demo", "prod"}:
            raise ValueError("KALSHI_ENV must be demo or prod")
        if self.mode not in {"collect", "paper", "live"}:
            raise ValueError("KALSHI_MODE must be collect, paper, or live")
        if self.mode == "live":
            if self.environment != "prod" or not self.allow_live or not confirm_live:
                raise RuntimeError(
                    "Live orders require KALSHI_ENV=prod, KALSHI_ALLOW_LIVE=true, "
                    "and the --confirm-live CLI flag"
                )
            if not self.api_key_id or not self.private_key_path:
                raise RuntimeError("Live orders require an API key ID and private-key path")
        if self.min_edge <= 0 or self.min_edge >= 1:
            raise ValueError("KALSHI_MIN_EDGE must be between 0 and 1")
        if min(self.max_order_risk, self.max_daily_loss, self.max_open_risk) <= 0:
            raise ValueError("Risk limits must be positive")

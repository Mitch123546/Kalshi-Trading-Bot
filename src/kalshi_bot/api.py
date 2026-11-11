from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa


class KalshiAPIError(RuntimeError):
    pass


class RequestSigner:
    def __init__(self, key_id: str, private_key_path: Path):
        self.key_id = key_id
        with private_key_path.open("rb") as handle:
            self.key = serialization.load_pem_private_key(handle.read(), password=None)

    def headers(self, method: str, path: str) -> dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        sign_path = path.split("?", 1)[0]
        message = f"{timestamp}{method.upper()}{sign_path}".encode()
        if isinstance(self.key, ed25519.Ed25519PrivateKey):
            signature = self.key.sign(message)
        elif isinstance(self.key, rsa.RSAPrivateKey):
            signature = self.key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=hashes.SHA256().digest_size,
                ),
                hashes.SHA256(),
            )
        else:
            raise TypeError("Kalshi supports Ed25519 or RSA private keys")
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }


class KalshiClient:
    def __init__(
        self,
        base_url: str,
        *,
        key_id: str | None = None,
        private_key_path: Path | None = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.base_path = "/trade-api/v2"
        self.signer = (
            RequestSigner(key_id, private_key_path) if key_id and private_key_path else None
        )
        self.http = httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": "kalshi-commodity-bot/0.1"}
        )

    async def __aenter__(self) -> KalshiClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self.http.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        authenticated: bool = False,
        retries: int = 4,
    ) -> dict[str, Any]:
        if not path.startswith("/"):
            path = "/" + path
        full_path = self.base_path + path
        headers: dict[str, str] = {}
        if authenticated:
            if not self.signer:
                raise RuntimeError("This request requires Kalshi API credentials")
            headers.update(self.signer.headers(method, full_path))
        if payload is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(retries + 1):
            response = await self.http.request(
                method, self.base_url + path, params=params, json=payload, headers=headers
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt == retries:
                break
            await __import__("asyncio").sleep(min(0.25 * (2**attempt), 4.0))
            if authenticated and self.signer:
                headers.update(self.signer.headers(method, full_path))
        if response.is_error:
            detail = response.text[:1000]
            raise KalshiAPIError(f"{method} {path}: HTTP {response.status_code}: {detail}")
        if not response.content:
            return {}
        return response.json()

    async def get_open_markets(self, series_ticker: str) -> list[dict[str, Any]]:
        data = await self.request(
            "GET",
            "/markets",
            params={"series_ticker": series_ticker, "status": "open", "limit": 1000},
        )
        return data.get("markets", [])

    async def get_market(self, ticker: str) -> dict[str, Any]:
        return (await self.request("GET", f"/markets/{ticker}"))["market"]

    async def get_orderbook(self, ticker: str, depth: int = 20) -> dict[str, Any]:
        data = await self.request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})
        return data.get("orderbook_fp") or data["orderbook"]

    async def get_event_live_data(
        self, event_ticker: str, chart_range: str = "15min"
    ) -> dict[str, Any]:
        return (
            await self.request(
                "GET", f"/live_data/events/{event_ticker}", params={"range": chart_range}
            )
        )["live_data"]

    async def get_series(self, ticker: str) -> dict[str, Any]:
        return (await self.request("GET", f"/series/{ticker}"))["series"]

    async def list_series(self, category: str = "Commodities") -> list[dict[str, Any]]:
        data = await self.request("GET", "/series", params={"category": category, "limit": 1000})
        return data.get("series", [])

    async def create_order_v2(
        self,
        *,
        ticker: str,
        side: str,
        count: int,
        price: float,
        client_order_id: str,
        post_only: bool = False,
        time_in_force: str = "immediate_or_cancel",
        exchange_index: int = -1,
    ) -> dict[str, Any]:
        payload = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,
            "count": f"{count:.2f}",
            "price": f"{price:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": post_only,
            "cancel_order_on_pause": True,
            "reduce_only": False,
            "subaccount": 0,
            "exchange_index": exchange_index,
        }
        return await self.request(
            "POST", "/portfolio/events/orders", payload=payload, authenticated=True
        )

    @staticmethod
    def canonical_query(params: dict[str, Any]) -> str:
        return urlencode(sorted((k, v) for k, v in params.items() if v is not None))

    @staticmethod
    def pretty(data: Any) -> str:
        return json.dumps(data, indent=2, sort_keys=True, default=str)

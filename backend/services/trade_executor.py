"""
Trade Executor — Dynamic Credentials
======================================
Updated to accept API credentials dynamically per request
instead of reading from environment variables at startup.

This allows multiple users to connect their own exchange accounts
through the dashboard without restarting the server.

Usage:
    # With dynamic credentials (from ConnectExchange flow):
    executor = TradeExecutor(api_key="xxx", api_secret="yyy",
                             exchange="binance", testnet=False)
    executor.connect()

    # Legacy mode (reads from settings/env — paper trading default):
    executor = TradeExecutor()
    executor.connect()   # LIVE_TRADING_ENABLED=False → paper mode
"""
import ccxt
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import backend.config.settings as settings

logger = logging.getLogger(__name__)

PERP_SYMBOL = settings.SYMBOL + ":USDT"

TESTNET_URLS = {
    "binance": {
        "public":  "https://testnet.binancefuture.com",
        "private": "https://testnet.binancefuture.com",
    },
    "bybit": {
        "public":  "https://api-testnet.bybit.com",
        "private": "https://api-testnet.bybit.com",
    },
}


class ExecutorError(Exception):
    pass


class TradeExecutor:

    def __init__(
        self,
        api_key:    Optional[str]  = None,
        api_secret: Optional[str]  = None,
        exchange:   Optional[str]  = None,
        testnet:    Optional[bool] = None,
    ):
        """
        Initialise executor.

        If api_key/api_secret are provided, they take priority over
        environment variables. This is the dynamic credentials path
        used when a user connects their account via the dashboard.

        If not provided, falls back to EXCHANGE_API_KEY/SECRET from
        settings (legacy env-var path, used for paper trading).
        """
        # Credential resolution: dynamic > env var
        self._api_key       = api_key    or settings.EXCHANGE_API_KEY    or ""
        self._api_secret    = api_secret or settings.EXCHANGE_API_SECRET or ""
        self._exchange_name = (exchange  or settings.EXCHANGE).lower()
        self._testnet       = testnet if testnet is not None else settings.EXCHANGE_TESTNET

        self._exchange:  Optional[ccxt.Exchange] = None
        self._connected: bool = False

        # Track whether this instance uses dynamic or env credentials
        self._dynamic = bool(api_key and api_secret)

    @property
    def paper_mode(self) -> bool:
        """True if paper trading is enabled via env vars and NO dynamic creds are provided."""
        return settings.PAPER_MODE and not self._dynamic

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Connect and verify exchange credentials.

        For paper trading (PAPER_MODE=True and no dynamic creds):
            Returns False immediately — no exchange connection needed.

        For live trading (dynamic creds provided OR PAPER_MODE=False):
            Attempts real exchange connection and verifies by fetching balance.
        """
        # Paper mode: no dynamic creds and paper mode enabled
        if self.paper_mode:
            logger.info("TradeExecutor: paper mode — no exchange connection.")
            return False

        if not self._api_key or not self._api_secret:
            logger.error(
                "TradeExecutor: no API credentials. "
                "Provide api_key/api_secret or set env vars."
            )
            return False

        try:
            exchange_cls = getattr(ccxt, self._exchange_name, None)
            if not exchange_cls:
                logger.error(f"Unknown exchange: {self._exchange_name}")
                return False

            config = {
                "apiKey":          self._api_key,
                "secret":          self._api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "recvWindow":  10000,    # ← add this — allows 10s clock drift
                },
            }

            if self._testnet:
                urls = TESTNET_URLS.get(self._exchange_name)
                if urls:
                    config["urls"] = {"api": urls}
                logger.info(f"Connecting to {self._exchange_name.upper()} TESTNET")
            else:
                logger.info(f"Connecting to {self._exchange_name.upper()} MAINNET")

            self._exchange = exchange_cls(config)
            self._exchange.load_markets()

            if self._exchange_name == "binance":
                self._ensure_one_way_mode()

            balance = self._exchange.fetch_balance()
            usdt    = float(balance.get("USDT", {}).get("free", 0))

            logger.info(
                f"Connected: {self._exchange_name.upper()} "
                f"({'TESTNET' if self._testnet else 'MAINNET'}) "
                f"USDT={usdt:,.2f}"
            )
            self._connected = True
            return True

        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _ensure_one_way_mode(self) -> None:
        try:
            resp     = self._exchange.fapiPrivateGetPositionSideDual()
            is_hedge = resp.get("dualSidePosition", False)
            if is_hedge:
                self._exchange.fapiPrivatePostPositionSideDual(
                    {"dualSidePosition": "false"}
                )
                logger.info("Binance: switched to one-way mode.")
        except Exception as e:
            logger.warning(f"Could not check Binance position mode: {e}")

    @property
    def connected(self) -> bool:
        return self._connected and self._exchange is not None

    @property
    def is_binance(self) -> bool:
        return self._exchange_name == "binance"
        
    def get_account_info(self) -> dict:
        """
        Fetch balance, leverage, margin mode, and open positions.
        Used for the dashboard exchange panel.
        """
        if not self._connected or not self._exchange:
            return {
                "balance": 0.0,
                "leverage": 1,
                "margin_mode": "cross",
                "open_positions": 0
            }

        try:
            balance = self._exchange.fetch_balance()
            usdt_balance = balance.get("USDT", {}).get("total", 0.0)

            positions = self._exchange.fetch_positions([PERP_SYMBOL]) if self._exchange.has.get("fetchPositions") else []
            active_positions = len([p for p in positions if float(p.get("info", {}).get("positionAmt", 0)) != 0])
            
            # Extract basic leverage/margin defaults; exchanges vary drastically
            leverage = 1
            margin_mode = "cross"
            
            if positions and len(positions) > 0:
                pos = positions[0]
                leverage = pos.get("leverage") or pos.get("info", {}).get("leverage") or 1
                margin_mode = "isolated" if pos.get("isolated") or pos.get("info", {}).get("isolated") else "cross"
                
            return {
                "balance": float(usdt_balance),
                "leverage": int(leverage),
                "margin_mode": str(margin_mode).lower(),
                "open_positions": active_positions
            }
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            return {
                "balance": 0.0,
                "leverage": 1,
                "margin_mode": "cross",
                "open_positions": 0
            }

    @property
    def is_bybit(self) -> bool:
        return self._exchange_name == "bybit"

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        if not self.connected:
            return 0.0
        try:
            bal = self._exchange.fetch_balance()
            return float(bal.get("USDT", {}).get("free", 0))
        except Exception as e:
            logger.error(f"get_balance failed: {e}")
            return 0.0

    def get_open_positions(self) -> list[dict]:
        if not self.connected:
            return []
        try:
            positions = self._exchange.fetch_positions([PERP_SYMBOL])
            result    = []
            for p in positions:
                size = float(p.get("contracts", 0) or 0)
                if size > 0:
                    result.append({
                        "symbol":            p.get("symbol"),
                        "side":              p.get("side"),
                        "size":              size,
                        "entry_price":       float(p.get("entryPrice",      0) or 0),
                        "mark_price":        float(p.get("markPrice",       0) or 0),
                        "unrealised_pnl":    float(p.get("unrealizedPnl",   0) or 0),
                        "percentage":        float(p.get("percentage",      0) or 0),
                        "liquidation_price": float(p.get("liquidationPrice",0) or 0),
                        "leverage":          float(p.get("leverage",        1) or 1),
                    })
            return result
        except Exception as e:
            logger.error(f"get_open_positions failed: {e}")
            return []

    # ── Order Placement ───────────────────────────────────────────────────────

    def place_order(
        self,
        direction:     str,
        position_size: float,
        entry_price:   float,
        sl_price:      float,
        tp_price:      float,
        signal_ts:     str,
    ) -> Optional[dict]:
        side = "buy" if direction == "BUY" else "sell"

        # Paper mode
        if self.paper_mode:
            logger.info(
                f"[PAPER] {direction} {position_size:.6f} {settings.SYMBOL} "
                f"entry~{entry_price:,.2f} SL={sl_price:,.2f} TP={tp_price:,.2f}"
            )
            return {
                "id":          f"paper_{int(time.time())}",
                "status":      "open",
                "paper":       True,
                "direction":   direction,
                "symbol":      settings.SYMBOL,
                "size":        position_size,
                "entry_price": entry_price,
                "sl_price":    sl_price,
                "tp_price":    tp_price,
                "signal_ts":   signal_ts,
                "opened_at":   datetime.now(timezone.utc).isoformat(),
                "exchange":    self._exchange_name,
            }

        if not self.connected:
            raise ExecutorError("Exchange not connected.")

        try:
            self._exchange.set_leverage(1, PERP_SYMBOL)
        except Exception:
            pass

        try:
            if self.is_binance:
                order = self._place_binance_order(
                    side, position_size, entry_price, sl_price, tp_price
                )
            else:
                order = self._place_bybit_order(
                    side, position_size, sl_price, tp_price
                )

            filled_price = float(order.get("average") or entry_price)
            logger.info(
                f"[LIVE] Order filled: id={order['id']} "
                f"price={filled_price:,.2f}"
            )

            return {
                "id":          order["id"],
                "status":      "open",
                "paper":       False,
                "direction":   direction,
                "symbol":      settings.SYMBOL,
                "size":        position_size,
                "entry_price": filled_price,
                "sl_price":    sl_price,
                "tp_price":    tp_price,
                "signal_ts":   signal_ts,
                "opened_at":   datetime.now(timezone.utc).isoformat(),
                "exchange":    self._exchange_name,
            }

        except ccxt.InsufficientFunds as e:
            raise ExecutorError(f"Insufficient funds: {e}") from e
        except ccxt.InvalidOrder as e:
            raise ExecutorError(f"Invalid order: {e}") from e
        except Exception as e:
            raise ExecutorError(f"Order failed: {e}") from e

    def _place_binance_order(
        self, side, position_size, entry_price, sl_price, tp_price
    ) -> dict:
        close_side  = "sell" if side == "buy" else "buy"
        entry_order = self._exchange.create_order(
            symbol = PERP_SYMBOL,
            type   = "market",
            side   = side,
            amount = position_size,
            params = {"positionSide": "BOTH"},
        )
        logger.info(f"Binance entry: {entry_order['id']}")

        for order_type, price, label in [
            ("stop_market",        sl_price, "SL"),
            ("take_profit_market", tp_price, "TP"),
        ]:
            try:
                o = self._exchange.create_order(
                    symbol = PERP_SYMBOL,
                    type   = order_type,
                    side   = close_side,
                    amount = position_size,
                    params = {
                        "stopPrice":    price,
                        "reduceOnly":   True,
                        "positionSide": "BOTH",
                        "workingType":  "MARK_PRICE",
                    },
                )
                logger.info(f"Binance {label}: {o['id']} @ {price}")
            except Exception as e:
                logger.error(f"Binance {label} order failed: {e}")

        return entry_order

    def _place_bybit_order(
        self, side, position_size, sl_price, tp_price
    ) -> dict:
        return self._exchange.create_order(
            symbol = PERP_SYMBOL,
            type   = "market",
            side   = side,
            amount = position_size,
            params = {
                "stopLoss":    {"triggerPrice": sl_price,  "type": "market"},
                "takeProfit":  {"triggerPrice": tp_price,  "type": "market"},
                "positionIdx": 0,
            },
        )

    def close_position(
        self, direction: str, size: float, reason: str = "manual"
    ) -> Optional[dict]:
        if self.paper_mode:
            return {"status": "closed", "paper": True, "reason": reason}

        if not self.connected:
            raise ExecutorError("Not connected.")

        close_side = "sell" if direction == "BUY" else "buy"
        try:
            if self.is_binance:
                try:
                    self._exchange.cancel_all_orders(PERP_SYMBOL)
                except Exception:
                    pass

            order = self._exchange.create_order(
                symbol = PERP_SYMBOL,
                type   = "market",
                side   = close_side,
                amount = size,
                params = {
                    "reduceOnly":   True,
                    "positionSide": "BOTH" if self.is_binance else None,
                    "positionIdx":  0      if self.is_bybit   else None,
                },
            )
            return {"status": "closed", "paper": False,
                    "order_id": order["id"], "reason": reason}
        except Exception as e:
            raise ExecutorError(f"Close failed: {e}") from e

    def get_closed_pnl(self, limit: int = 50) -> list[dict]:
        if not self.connected:
            return []
        try:
            records = self._exchange.fetch_my_trades(PERP_SYMBOL, limit=limit)
            return [
                {
                    "order_id":  r.get("order"),
                    "symbol":    r.get("symbol"),
                    "side":      r.get("side"),
                    "price":     float(r.get("price",  0)),
                    "amount":    float(r.get("amount", 0)),
                    "pnl":       float(r.get("info", {}).get(
                                    "realizedPnl" if self.is_binance
                                    else "closedPnl", 0)),
                    "fee":       float(r.get("fee", {}).get("cost", 0)),
                    "timestamp": r.get("datetime"),
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"get_closed_pnl failed: {e}")
            return []
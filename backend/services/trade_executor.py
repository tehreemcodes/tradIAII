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


class ExecutorError(Exception):
    pass


class TradeExecutor:

    def __init__(
        self,
        api_key:    Optional[str]  = None,
        api_secret: Optional[str]  = None,
        exchange:   Optional[str]  = None,
        testnet:    Optional[bool] = None,
        leverage:   Optional[int]  = None,
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
        self._leverage      = leverage if leverage is not None else settings.DEFAULT_LEVERAGE

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

           self._exchange = exchange_cls(config)

            if self._testnet:
                self._exchange.set_sandbox_mode(True)
                logger.info(f"Connecting to {self._exchange_name.upper()} TESTNET")
            else:
                logger.info(f"Connecting to {self._exchange_name.upper()} MAINNET")

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

    # ── Pre-Order Margin Validation ──────────────────────────────────────────

    def _validate_margin(
        self, position_size: float, entry_price: float, side: str
    ) -> float:
        """
        Validate that the position fits within available margin.

        If required_margin > available_balance, scale down position_size
        so it fits (with a 5% safety buffer).  Returns the (possibly
        reduced) position_size, or 0.0 if even the minimum trade is
        impossible.
        """
        MIN_NOTIONAL = 5.0   # Binance Futures minimum notional (USD)
        MARGIN_BUFFER = 0.95  # use at most 95% of available balance

        try:
            available = self.get_balance()
        except Exception as e:
            logger.error(f"_validate_margin: could not fetch balance: {e}")
            return position_size   # proceed optimistically

        notional        = position_size * entry_price
        required_margin = notional / self._leverage

        logger.info(
            f"[MARGIN CHECK] size={position_size:.6f} | "
            f"notional=${notional:,.2f} | leverage={self._leverage}x | "
            f"required_margin=${required_margin:,.2f} | "
            f"available=${available:,.2f}"
        )

        if required_margin <= available * MARGIN_BUFFER:
            return position_size   # fits fine

        # Scale down to fit within available margin
        max_notional      = available * MARGIN_BUFFER * self._leverage
        adjusted_size     = max_notional / entry_price

        adjusted_notional = adjusted_size * entry_price
        if adjusted_notional < MIN_NOTIONAL:
            logger.error(
                f"[MARGIN CHECK] Scaled-down notional ${adjusted_notional:,.2f} "
                f"is below Binance minimum ${MIN_NOTIONAL}. Trade impossible."
            )
            return 0.0

        logger.warning(
            f"[MARGIN CHECK] Position scaled down: "
            f"{position_size:.6f} → {adjusted_size:.6f} "
            f"(notional ${notional:,.2f} → ${adjusted_notional:,.2f}) "
            f"to fit available margin ${available:,.2f}"
        )
        return adjusted_size

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

        # ── Set leverage (configurable, default 20x) ─────────────────────────
        try:
            self._exchange.set_leverage(self._leverage, PERP_SYMBOL)
            logger.info(f"Setting leverage to {self._leverage}x for {PERP_SYMBOL}")
        except Exception as e:
            logger.warning(f"Could not set leverage to {self._leverage}x: {e}")

        # ── Pre-order margin validation & auto scale-down ────────────────────
        position_size = self._validate_margin(
            position_size, entry_price, side
        )
        if position_size <= 0:
            raise ExecutorError(
                f"Position size too small after margin validation. "
                f"Balance may be insufficient for minimum trade."
            )

        try:
            if self.is_binance:
                order = self._place_binance_order(
                    side, position_size, entry_price, sl_price, tp_price
                )
            else:
                order = self._place_bybit_order(
                    side, position_size, sl_price, tp_price
                )

            filled_price = float(order.get("average") or order.get("price") or entry_price)
            logger.info(
                f"[LIVE] Order ({settings.ORDER_TYPE}): id={order['id']} "
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
                "order_type":  settings.ORDER_TYPE,
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
        order_type  = settings.ORDER_TYPE.lower()
        
        # Apply slight offset to limit price to improve fill chance (0.01%)
        # If BUY: slightly higher, if SELL: slightly lower than structural entry
        offset = 1.0001 if side == "buy" else 0.9999
        limit_price = round(entry_price * offset, 2)

        params = {"positionSide": "BOTH"}
        if order_type == "limit":
            params["timeInForce"] = "GTC"

        entry_order = self._exchange.create_order(
            symbol = PERP_SYMBOL,
            type   = order_type,
            side   = side,
            amount = position_size,
            price  = limit_price if order_type == "limit" else None,
            params = params,
        )
        logger.info(f"Binance entry ({order_type}): {entry_order['id']} @ {limit_price if order_type == 'limit' else 'MARKET'}")

        for o_type, price, label in [
            ("stop_market",        sl_price, "SL"),
            ("take_profit_market", tp_price, "TP"),
        ]:
            try:
                o = self._exchange.create_order(
                    symbol = PERP_SYMBOL,
                    type   = o_type,
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

    def cancel_all_conditional_orders(self, symbol: Optional[str] = None) -> None:
        """Cancel all open orders (SL/TP triggers) for the symbol."""
        if self.paper_mode or not self.connected:
            return
        
        target = symbol or PERP_SYMBOL
        # Support both 'BTCUSDT' or 'BTC/USDT:USDT' formats
        if ":" not in target and self.is_binance:
            target = target.replace("/", "") + ":USDT"

        try:
            if self.is_binance:
                self._exchange.cancel_all_orders(target)
                logger.info(f"Cancelled all conditional orders for {target}")
        except Exception as e:
            logger.warning(f"Failed to cancel conditional orders for {target}: {e}")

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
                    "timestamp": r.get("timestamp"), # raw ms
                    "datetime":  r.get("datetime"),  # ISO string
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"get_closed_pnl failed: {e}")
            return []

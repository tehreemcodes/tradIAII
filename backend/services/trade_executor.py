"""
Trade Executor -- Native Binance Futures API
=============================================
Always connects to demo-fapi.binance.com (Binance Futures Demo).
BINANCE_BASE_URL is enforced at module load via assertion in settings.

Key Production Features:
- BINANCE_BASE_URL always used -- no conditional testnet/mainnet split
- LOT_SIZE stepSize + PRICE_FILTER tickSize loaded from /fapi/v1/exchangeInfo on startup
- SL/TP orders use closePosition="true" (fixes Binance -4120 error)
- Emergency close: 3 retries with 1s sleep, CRITICAL log on every failure
- Limit entry with book ticker for tighter fills (USE_LIMIT_ENTRY=True)
- Idempotency via newClientOrderId
- Automatic serverTime synchronization to prevent -1021 timestamp drift
"""
import logging
import time
import math
import hmac
import hashlib
import uuid
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import Timeout
from urllib3.util.retry import Retry
from urllib.parse import urlencode
from datetime import datetime, timezone
from typing import Optional

import backend.config.settings as settings
from backend.config.settings import (
    BINANCE_BASE_URL,
    USE_LIMIT_ENTRY,
    ENTRY_LIMIT_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)

PERP_SYMBOL = settings.SYMBOL + ":USDT"


class ExecutorError(Exception):
    pass


class BinanceFuturesExecutor:
    def __init__(
        self,
        api_key:    Optional[str]  = None,
        api_secret: Optional[str]  = None,
        exchange:   Optional[str]  = None,  # kept for API signature compat, ignored
        testnet:    Optional[bool] = None,  # kept for API signature compat, ignored
        leverage:   Optional[int]  = None,
    ):
        self._api_key    = api_key    or settings.EXCHANGE_API_KEY    or ""
        self._api_secret = api_secret or settings.EXCHANGE_API_SECRET or ""
        self._leverage   = leverage if leverage is not None else settings.DEFAULT_LEVERAGE

        self._connected: bool = False
        self._dynamic = bool(api_key and api_secret)
        self._time_offset = 0

        # Always use BINANCE_BASE_URL -- no conditional testnet/mainnet logic
        self.base_url = BINANCE_BASE_URL

        self.symbol = settings.SYMBOL.replace("/", "")

        # Exchange info -- loaded on connect(); safe defaults for BTC
        self._lot_step: float  = 0.001   # LOT_SIZE stepSize
        self._tick_size: float = 0.10    # PRICE_FILTER tickSize

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    @property
    def paper_mode(self) -> bool:
        return settings.PAPER_MODE and not self._dynamic

    # -- Authentication & Requests -----------------------------------------

    def _sync_time(self):
        try:
            res = self._request("GET", "/fapi/v1/time", signed=False)
            server_time = res["serverTime"]
            local_time = int(time.time() * 1000)
            self._time_offset = server_time - local_time
            logger.info(f"Server time synced. Offset: {self._time_offset}ms")
        except Exception as e:
            logger.warning(f"Could not sync server time: {e}")

    def _get_timestamp(self):
        return int(time.time() * 1000) + self._time_offset

    def _sign(self, query_string: str) -> str:
        return hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method:   str,
        endpoint: str,
        params:   dict = None,
        signed:   bool = True,
        timeout:  int  = 10,
    ):
        url    = self.base_url + endpoint
        params = params or {}

        if signed:
            params["timestamp"]  = self._get_timestamp()
            params["recvWindow"] = 10000
            query_string = urlencode(params)
            signature    = self._sign(query_string)
            query_string += f"&signature={signature}"
            full_url = f"{url}?{query_string}"
            headers  = {"X-MBX-APIKEY": self._api_key}
        else:
            query_string = urlencode(params)
            full_url = f"{url}?{query_string}" if query_string else url
            headers  = {}

        try:
            if method == "GET":
                response = self.session.get(full_url, headers=headers, timeout=timeout)
            elif method == "POST":
                response = self.session.post(full_url, headers=headers, timeout=timeout)
            elif method == "DELETE":
                response = self.session.delete(full_url, headers=headers, timeout=timeout)
            elif method == "PUT":
                response = self.session.put(full_url, headers=headers, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            data = response.json()
            if response.status_code != 200:
                code = data.get("code")
                msg  = data.get("msg")
                raise ExecutorError(f"API Error [{code}]: {msg}")
            return data
        except ExecutorError:
            raise
        except Timeout:
            raise Timeout("Request timed out")
        except Exception as e:
            raise ExecutorError(f"HTTP Request failed: {e}")

    # -- Exchange Info -------------------------------------------------------

    def _load_exchange_info(self):
        """
        Load LOT_SIZE stepSize and PRICE_FILTER tickSize from Binance exchangeInfo.
        Called on connect() so _round_qty() / _round_price() use correct precision.
        """
        try:
            info = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
            for sym_info in info.get("symbols", []):
                if sym_info["symbol"] == self.symbol:
                    for f in sym_info.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            self._lot_step = float(f["stepSize"])
                        elif f["filterType"] == "PRICE_FILTER":
                            self._tick_size = float(f["tickSize"])
            logger.info(
                f"[EXCHANGE INFO] {self.symbol}: "
                f"lot_step={self._lot_step}  tick_size={self._tick_size}"
            )
        except Exception as e:
            logger.warning(f"Failed to load exchangeInfo, using defaults: {e}")

    def _round_qty(self, qty: float) -> float:
        """Round quantity DOWN to the nearest LOT_SIZE stepSize."""
        step = self._lot_step
        if step <= 0:
            return round(qty, 3)
        precision = max(0, round(-math.log10(step)))
        return round(math.floor(qty / step) * step, precision)

    def _round_price(self, price: float) -> str:
        """Round price DOWN to PRICE_FILTER tickSize; return as string for API."""
        tick = self._tick_size
        if tick <= 0:
            return str(round(price, 2))
        precision = max(0, round(-math.log10(tick)))
        rounded   = round(math.floor(price / tick) * tick, precision)
        return f"{rounded:.{precision}f}"

    # -- Connection ---------------------------------------------------------

    def connect(self) -> bool:
        if self.paper_mode:
            logger.info("BinanceFuturesExecutor: paper mode -- no exchange connection.")
            return False

        if not self._api_key or not self._api_secret:
            logger.error("BinanceFuturesExecutor: no API credentials provided.")
            return False

        try:
            self._sync_time()
            self._load_exchange_info()
            pos_mode = self._ensure_one_way_mode()
            self._set_leverage()

            logger.info(
                f"[STARTUP] CONFIRMED base_url={self.base_url} "
                f"symbol={self.symbol} leverage={self._leverage}x "
                f"lot_step={self._lot_step} tick_size={self._tick_size} "
                f"position_mode={pos_mode}"
            )

            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _ensure_one_way_mode(self) -> str:
        try:
            res     = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
            is_dual = res.get("dualSidePosition", False)
            if is_dual:
                self._request(
                    "POST", "/fapi/v1/positionSide/dual",
                    {"dualSidePosition": "false"}, signed=True,
                )
                logger.info("[STARTUP] CONFIRMED Binance: switched to one-way mode.")
                return "one-way (switched)"
            else:
                logger.info("[STARTUP] CONFIRMED Binance: already in one-way mode.")
                return "one-way (already set)"
        except Exception as e:
            logger.warning(f"Could not enforce one-way mode: {e}")
            return "unknown"

    def _set_leverage(self):
        try:
            self._request("POST", "/fapi/v1/leverage", {
                "symbol":   self.symbol,
                "leverage": self._leverage,
            }, signed=True)
            logger.info(f"Set leverage to {self._leverage}x for {self.symbol}")
        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_binance(self) -> bool:
        return True

    @property
    def is_bybit(self) -> bool:
        return False

    # -- Account Info -------------------------------------------------------

    def get_balance(self) -> float:
        if not self.connected and not self.paper_mode:
            return 0.0
        if self.paper_mode:
            return settings.INITIAL_CAPITAL
        try:
            res = self._request("GET", "/fapi/v2/balance", signed=True)
            for asset in res:
                if asset["asset"] == "USDT":
                    return float(asset["availableBalance"])
            return 0.0
        except Exception as e:
            logger.error(f"get_balance failed: {e}")
            return 0.0

    def get_account_info(self) -> dict:
        if not self.connected:
            return {"balance": 0.0, "leverage": 1, "margin_mode": "cross", "open_positions": 0}

        try:
            bal_res    = self._request("GET", "/fapi/v2/account", signed=True)
            usdt_total = float(bal_res.get("totalWalletBalance", 0.0))

            pos_res        = self._request("GET", "/fapi/v2/positionRisk", {"symbol": self.symbol}, signed=True)
            active_positions = 0
            leverage         = self._leverage
            margin_mode      = "cross"

            if pos_res and len(pos_res) > 0:
                pos = pos_res[0]
                if float(pos.get("positionAmt", 0)) != 0:
                    active_positions = 1
                leverage    = int(pos.get("leverage", leverage))
                margin_mode = "isolated" if pos.get("marginType") == "isolated" else "cross"

            return {
                "balance":        usdt_total,
                "leverage":       leverage,
                "margin_mode":    margin_mode,
                "open_positions": active_positions,
            }
        except Exception as e:
            logger.error(f"get_account_info failed: {e}")
            return {"balance": 0.0, "leverage": 1, "margin_mode": "cross", "open_positions": 0}

    def get_open_positions(self) -> list[dict]:
        if not self.connected:
            return []
        try:
            pos_res = self._request("GET", "/fapi/v2/positionRisk", {"symbol": self.symbol}, signed=True)
            result  = []
            for p in pos_res:
                amt = float(p.get("positionAmt", 0))
                if amt != 0:
                    side = "long" if amt > 0 else "short"
                    result.append({
                        "symbol":            getattr(settings, "SYMBOL", p.get("symbol")),
                        "side":              side,
                        "size":              abs(amt),
                        "entry_price":       float(p.get("entryPrice", 0)),
                        "mark_price":        float(p.get("markPrice", 0)),
                        "unrealised_pnl":    float(p.get("unRealizedProfit", 0)),
                        "liquidation_price": float(p.get("liquidationPrice", 0)),
                        "leverage":          float(p.get("leverage", 1)),
                    })
            return result
        except Exception as e:
            logger.error(f"get_open_positions failed: {e}")
            return []

    # -- Order Execution ----------------------------------------------------

    def _validate_margin(self, position_size: float, entry_price: float) -> float:
        MIN_NOTIONAL  = 5.0
        MARGIN_BUFFER = 0.95

        try:
            available = self.get_balance()
        except Exception:
            return position_size

        notional        = position_size * entry_price
        required_margin = notional / self._leverage

        if required_margin <= available * MARGIN_BUFFER:
            return position_size

        max_notional  = available * MARGIN_BUFFER * self._leverage
        adjusted_size = max_notional / entry_price

        if adjusted_size * entry_price < MIN_NOTIONAL:
            return 0.0
        return self._round_qty(adjusted_size)

    def _get_book_ticker(self) -> dict:
        """Fetch best bid/ask from Binance book ticker."""
        return self._request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": self.symbol}, signed=False)

    def _safe_place_entry_order(self, params: dict):
        """Place entry order; on Timeout, query by newClientOrderId before retrying once."""
        client_id = params["newClientOrderId"]

        try:
            return self._request("POST", "/fapi/v1/order", params, signed=True)
        except Timeout:
            logger.warning(f"Timeout placing entry order. Checking if {client_id} resolved...")
            try:
                order_status = self._request("GET", "/fapi/v1/order", {
                    "symbol":            self.symbol,
                    "origClientOrderId": client_id,
                }, signed=True)
                logger.info(f"Order found on query: {order_status.get('status')} - DO NOT RETRY.")
                return order_status
            except ExecutorError as e:
                if "Code -2013" in str(e) or "doesn't exist" in str(e).lower():
                    logger.warning("Order not found on Binance. Retrying once...")
                    return self._request("POST", "/fapi/v1/order", params, signed=True)
                else:
                    raise ExecutorError(f"Verification failed: {e}")

    def place_order(
        self,
        direction:     str,
        position_size: float,
        entry_price:   float,
        sl_price:      float,
        tp_price:      float,
        signal_ts:     str,
        strategy_type: str = "legacy",
    ) -> Optional[dict]:
        side       = "BUY"  if direction.upper() == "BUY" else "SELL"
        close_side = "SELL" if side == "BUY" else "BUY"

        if self.paper_mode:
            logger.info(f"[PAPER] {direction} {position_size:.6f} {settings.SYMBOL} entry~{entry_price:,.2f}")
            return {
                "id":           f"paper_{int(time.time())}",
                "status":       "open",
                "paper":        True,
                "direction":    direction,
                "symbol":       settings.SYMBOL,
                "size":         position_size,
                "entry_price":  entry_price,
                "sl_price":     sl_price,
                "tp_price":     tp_price,
                "signal_ts":    signal_ts,
                "opened_at":    datetime.now(timezone.utc).isoformat(),
                "exchange":     "binance",
                "order_type":   "LIMIT" if USE_LIMIT_ENTRY else "MARKET",
                "strategy_type": strategy_type,
            }

        if not self.connected:
            raise ExecutorError("Not connected.")

        position_size = self._validate_margin(position_size, entry_price)
        if position_size <= 0:
            raise ExecutorError("Position size too small after margin validation.")

        qty = self._round_qty(position_size)
        if qty <= 0:
            raise ExecutorError("Rounded quantity is zero -- lot_step too large for this position.")

        # -- 1. Entry Order -----------------------------------------------
        strat_prefix     = strategy_type[:5] if strategy_type else "legcy"
        unique_client_id = f"tradia_{strat_prefix}_{uuid.uuid4().hex[:10]}"

        if USE_LIMIT_ENTRY:
            # Fetch book ticker for tighter entry price
            try:
                ticker      = self._get_book_ticker()
                limit_price = float(ticker["askPrice"] if side == "BUY" else ticker["bidPrice"])
            except Exception:
                limit_price = entry_price   # fallback to signal close price

            params = {
                "symbol":           self.symbol,
                "side":             side,
                "type":             "LIMIT",
                "quantity":         qty,
                "price":            self._round_price(limit_price),
                "timeInForce":      "GTC",
                "positionSide":     "BOTH",
                "newClientOrderId": unique_client_id,
            }
            try:
                entry_res = self._safe_place_entry_order(params)
                logger.info(f"LIMIT entry placed: {entry_res.get('orderId')} status={entry_res.get('status')}")

                # Poll for fill until ENTRY_LIMIT_TIMEOUT_SEC
                deadline = time.time() + ENTRY_LIMIT_TIMEOUT_SEC
                while time.time() < deadline:
                    chk = self._request("GET", "/fapi/v1/order", {
                        "symbol":  self.symbol,
                        "orderId": entry_res.get("orderId"),
                    }, signed=True)
                    status       = chk.get("status")
                    executed_qty = float(chk.get("executedQty", 0))

                    if status == "FILLED":
                        entry_res = chk
                        logger.info(f"LIMIT entry FILLED: qty={executed_qty}")
                        break
                    elif status in ("CANCELED", "EXPIRED", "REJECTED"):
                        raise ExecutorError(f"Limit entry order {status}")
                    time.sleep(2)
                else:
                    # Timeout: cancel and fall back to MARKET
                    try:
                        self._request("DELETE", "/fapi/v1/order", {
                            "symbol":  self.symbol,
                            "orderId": entry_res.get("orderId"),
                        }, signed=True)
                    except Exception:
                        pass
                    logger.warning(
                        f"Limit entry timed out after {ENTRY_LIMIT_TIMEOUT_SEC}s "
                        f"-- cancelled, falling back to MARKET"
                    )
                    mkt_id     = f"tradia_mkt_{uuid.uuid4().hex[:10]}"
                    mkt_params = {
                        "symbol":           self.symbol,
                        "side":             side,
                        "type":             "MARKET",
                        "quantity":         qty,
                        "positionSide":     "BOTH",
                        "newClientOrderId": mkt_id,
                    }
                    entry_res = self._safe_place_entry_order(mkt_params)
                    logger.info(f"MARKET fallback entry: {entry_res.get('orderId')}")

            except ExecutorError:
                raise
            except Exception as e:
                raise ExecutorError(f"Limit entry failed: {e}")

        else:
            # Pure market order
            params = {
                "symbol":           self.symbol,
                "side":             side,
                "type":             "MARKET",
                "quantity":         qty,
                "positionSide":     "BOTH",
                "newClientOrderId": unique_client_id,
            }
            try:
                entry_res = self._safe_place_entry_order(params)
                logger.info(
                    f"MARKET entry resolved: {entry_res.get('orderId')} "
                    f"status={entry_res.get('status')}"
                )
            except Exception as e:
                raise ExecutorError(f"Entry Order failed completely: {e}")

        # -- 2. SL/TP using closePosition='true' (fixes Binance -4120) -----
        try:
            sl_rounded = self._round_price(sl_price)
            tp_rounded = self._round_price(tp_price)

            # Stop Loss
            self._request("POST", "/fapi/v1/order", {
                "symbol":       self.symbol,
                "side":         close_side,
                "type":         "STOP_MARKET",
                "stopPrice":    sl_rounded,
                "closePosition": "true",
                "timeInForce":  "GTE_GTC",
                "workingType":  "CONTRACT_PRICE",
                "positionSide": "BOTH",
            }, signed=True)
            logger.info(f"SL placed at {sl_rounded}")

            # Take Profit
            self._request("POST", "/fapi/v1/order", {
                "symbol":       self.symbol,
                "side":         close_side,
                "type":         "TAKE_PROFIT_MARKET",
                "stopPrice":    tp_rounded,
                "closePosition": "true",
                "timeInForce":  "GTE_GTC",
                "workingType":  "CONTRACT_PRICE",
                "positionSide": "BOTH",
            }, signed=True)
            logger.info(f"TP placed at {tp_rounded}")

            executed_qty = float(entry_res.get("executedQty", qty))

            return {
                "id":           str(entry_res.get("orderId")),
                "status":       "open",
                "paper":        False,
                "direction":    direction,
                "symbol":       settings.SYMBOL,
                "size":         executed_qty if executed_qty > 0 else qty,
                "entry_price":  float(entry_res.get("avgPrice") or entry_res.get("price") or entry_price),
                "sl_price":     sl_price,
                "tp_price":     tp_price,
                "signal_ts":    signal_ts,
                "opened_at":    datetime.now(timezone.utc).isoformat(),
                "exchange":     "binance",
                "order_type":   "LIMIT" if USE_LIMIT_ENTRY else "MARKET",
                "strategy_type": strategy_type,
            }

        except Exception as e:
            logger.error(f"Failed to place SL/TP: {e}")
            executed_qty = float(entry_res.get("executedQty", 0)) if "entry_res" in dir() else qty
            if executed_qty > 0:
                logger.critical(
                    f"UNPROTECTED POSITION DETECTED! direction={direction} qty={executed_qty} "
                    f"-- attempting emergency close"
                )
                self._emergency_market_close(direction, executed_qty, reason="sltp_placement_failed")
            raise ExecutorError(f"SL/TP Placement failed: {e}")

    def _emergency_market_close(self, direction: str, size: float, reason: str = "emergency") -> None:
        """
        Emergency market close with 3 retries and 1s sleep between attempts.
        Raises ExecutorError if all 3 attempts fail.
        """
        close_side = "SELL" if direction.upper() == "BUY" else "BUY"
        qty        = self._round_qty(size)
        last_exc   = None

        for attempt in range(1, 4):
            try:
                self.cancel_all_conditional_orders()
                res = self._request("POST", "/fapi/v1/order", {
                    "symbol":     self.symbol,
                    "side":       close_side,
                    "type":       "MARKET",
                    "quantity":   qty,
                    "reduceOnly": "true",
                }, signed=True)
                logger.critical(
                    f"[EMERGENCY CLOSE] SUCCESS attempt={attempt}/3 "
                    f"reason={reason} orderId={res.get('orderId')}"
                )
                return
            except Exception as e:
                last_exc = e
                logger.critical(
                    f"[EMERGENCY CLOSE] FAILED attempt={attempt}/3 "
                    f"reason={reason} error={e}"
                )
                if attempt < 3:
                    time.sleep(1)

        raise ExecutorError(
            f"[EMERGENCY CLOSE] ALL 3 ATTEMPTS FAILED reason={reason}: {last_exc}"
        )

    def close_position(self, direction: str, size: float, reason: str = "manual") -> Optional[dict]:
        if self.paper_mode:
            return {"status": "closed", "paper": True, "reason": reason}

        if not self.connected:
            raise ExecutorError("Not connected.")

        close_side = "SELL" if direction.upper() == "BUY" else "BUY"
        qty        = self._round_qty(size)

        try:
            self.cancel_all_conditional_orders()
            res = self._request("POST", "/fapi/v1/order", {
                "symbol":     self.symbol,
                "side":       close_side,
                "type":       "MARKET",
                "quantity":   qty,
                "reduceOnly": "true",
            }, signed=True)
            return {
                "status":   "closed",
                "paper":    False,
                "order_id": str(res.get("orderId")),
                "reason":   reason,
            }
        except Exception as e:
            raise ExecutorError(f"Close failed: {e}")

    def cancel_all_conditional_orders(self, symbol: Optional[str] = None) -> None:
        """Cancel only STOP_MARKET and TAKE_PROFIT_MARKET orders; leave entry orders untouched."""
        if self.paper_mode or not self.connected:
            return
        sym = symbol.replace("/", "").replace(":USDT", "") if symbol else self.symbol
        try:
            orders       = self._request("GET", "/fapi/v1/openOrders", {"symbol": sym}, signed=True)
            cancel_count = 0
            for order in orders:
                if order.get("type") in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
                    self._request("DELETE", "/fapi/v1/order", {
                        "symbol":  sym,
                        "orderId": order["orderId"],
                    }, signed=True)
                    cancel_count += 1
            logger.info(f"Cancelled {cancel_count} conditional SL/TP orders.")
        except Exception as e:
            logger.warning(f"Failed to cleanly cancel conditional orders: {e}")

    def get_closed_pnl(self, limit: int = 50) -> list[dict]:
        if not self.connected:
            return []
        try:
            res    = self._request("GET", "/fapi/v1/userTrades", {"symbol": self.symbol, "limit": limit}, signed=True)
            result = []
            for r in res:
                result.append({
                    "order_id":  r.get("orderId"),
                    "symbol":    r.get("symbol"),
                    "side":      str(r.get("side")).lower(),
                    "price":     float(r.get("price", 0)),
                    "amount":    float(r.get("qty", 0)),
                    "pnl":       float(r.get("realizedPnl", 0)),
                    "fee":       float(r.get("commission", 0)),
                    "timestamp": r.get("time"),
                    "datetime":  datetime.fromtimestamp(
                        r.get("time", 0) / 1000, tz=timezone.utc
                    ).isoformat(),
                })
            return result
        except Exception as e:
            logger.error(f"get_closed_pnl failed: {e}")
            return []


TradeExecutor = BinanceFuturesExecutor

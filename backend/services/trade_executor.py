"""
Trade Executor \u2014 Native Binance Futures API
=============================================
Replaces CCXT with native REST endpoints for robust execution.
Supports both fapi.binance.com (Mainnet) and demo-fapi.binance.com (Testnet).

Key Production Features:
- Idempotency via newClientOrderId
- Automatic serverTime synchronization to prevent -1021 timestamp drift
- Dynamic Partial Fill handling via executedQty
- Safe Error handling and timeout recovery
"""
import logging
import time
import hmac
import hashlib
import uuid
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import Timeout, RequestException
from urllib3.util.retry import Retry
from urllib.parse import urlencode
from datetime import datetime, timezone
from typing import Optional

import backend.config.settings as settings

logger = logging.getLogger(__name__)

PERP_SYMBOL = settings.SYMBOL + ":USDT"


class ExecutorError(Exception):
    pass

class BinanceFuturesExecutor:
    def __init__(
        self,
        api_key:    Optional[str]  = None,
        api_secret: Optional[str]  = None,
        exchange:   Optional[str]  = None, # Kept for API signature compatibility, ignored
        testnet:    Optional[bool] = None,
        leverage:   Optional[int]  = None,
    ):
        self._api_key       = api_key    or settings.EXCHANGE_API_KEY    or ""
        self._api_secret    = api_secret or settings.EXCHANGE_API_SECRET or ""
        self._testnet       = testnet if testnet is not None else settings.EXCHANGE_TESTNET
        self._leverage      = leverage if leverage is not None else settings.DEFAULT_LEVERAGE

        self._connected: bool = False
        self._dynamic = bool(api_key and api_secret)
        self._time_offset = 0

        if self._testnet:
            self.base_url = "https://demo-fapi.binance.com"
        else:
            self.base_url = "https://fapi.binance.com"

        self.symbol = settings.SYMBOL.replace("/", "")

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 503, 500, 502, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    @property
    def paper_mode(self) -> bool:
        return settings.PAPER_MODE and not self._dynamic

    # \u2500\u2500 Authentication & Requests \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

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
            self._api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = True, timeout: int = 10):
        url = self.base_url + endpoint
        params = params or {}

        if signed:
            params['timestamp'] = self._get_timestamp()
            params['recvWindow'] = 10000
            query_string = urlencode(params)
            signature = self._sign(query_string)
            query_string += f"&signature={signature}"
            full_url = f"{url}?{query_string}"
            headers = {"X-MBX-APIKEY": self._api_key}
        else:
            query_string = urlencode(params)
            full_url = f"{url}?{query_string}" if query_string else url
            headers = {}

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
                code = data.get('code')
                msg = data.get('msg')
                raise ExecutorError(f"API Error [{code}]: {msg}")
            return data
        except ExecutorError:
            raise
        except Timeout:
            raise Timeout("Request timed out")
        except Exception as e:
            raise ExecutorError(f"HTTP Request failed: {e}")

    # \u2500\u2500 Connection \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def connect(self) -> bool:
        if self.paper_mode:
            logger.info("BinanceFuturesExecutor: paper mode \u2014 no exchange connection.")
            return False

        if not self._api_key or not self._api_secret:
            logger.error("BinanceFuturesExecutor: no API credentials provided.")
            return False

        try:
            self._sync_time()
            self._ensure_one_way_mode()
            self._set_leverage()

            logger.info(
                f"Connected to Binance Futures "
                f"({'TESTNET' if self._testnet else 'MAINNET'})"
            )

            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _ensure_one_way_mode(self):
        try:
            res = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
            is_dual = res.get("dualSidePosition", False)
            if is_dual:
                self._request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "false"}, signed=True)
                logger.info("Binance: switched to one-way mode.")
        except Exception as e:
            logger.warning(f"Could not enforce one-way mode: {e}")

    def _set_leverage(self):
        try:
            self._request("POST", "/fapi/v1/leverage", {
                "symbol": self.symbol,
                "leverage": self._leverage
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

    # \u2500\u2500 Account Info \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

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
            bal_res = self._request("GET", "/fapi/v2/account", signed=True)
            usdt_total = float(bal_res.get("totalWalletBalance", 0.0))

            pos_res = self._request("GET", "/fapi/v2/positionRisk", {"symbol": self.symbol}, signed=True)
            active_positions = 0
            leverage = self._leverage
            margin_mode = "cross"

            if pos_res and len(pos_res) > 0:
                pos = pos_res[0]
                if float(pos.get("positionAmt", 0)) != 0:
                    active_positions = 1
                leverage = int(pos.get("leverage", leverage))
                margin_mode = "isolated" if pos.get("marginType") == "isolated" else "cross"
            
            return {
                "balance": usdt_total,
                "leverage": leverage,
                "margin_mode": margin_mode,
                "open_positions": active_positions
            }
        except Exception as e:
            logger.error(f"get_account_info failed: {e}")
            return {"balance": 0.0, "leverage": 1, "margin_mode": "cross", "open_positions": 0}

    def get_open_positions(self) -> list[dict]:
        if not self.connected:
            return []
        try:
            pos_res = self._request("GET", "/fapi/v2/positionRisk", {"symbol": self.symbol}, signed=True)
            result = []
            for p in pos_res:
                amt = float(p.get("positionAmt", 0))
                if amt != 0:
                    side = "long" if amt > 0 else "short"
                    result.append({
                        "symbol": getattr(settings, 'SYMBOL', p.get("symbol")),
                        "side": side,
                        "size": abs(amt),
                        "entry_price": float(p.get("entryPrice", 0)),
                        "mark_price": float(p.get("markPrice", 0)),
                        "unrealised_pnl": float(p.get("unRealizedProfit", 0)),
                        "liquidation_price": float(p.get("liquidationPrice", 0)),
                        "leverage": float(p.get("leverage", 1))
                    })
            return result
        except Exception as e:
            logger.error(f"get_open_positions failed: {e}")
            return []

    # \u2500\u2500 Order Execution \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _validate_margin(self, position_size: float, entry_price: float) -> float:
        MIN_NOTIONAL = 5.0
        MARGIN_BUFFER = 0.95

        try:
            available = self.get_balance()
        except:
            return position_size

        notional = position_size * entry_price
        required_margin = notional / self._leverage

        if required_margin <= available * MARGIN_BUFFER:
            return position_size

        max_notional = available * MARGIN_BUFFER * self._leverage
        adjusted_size = max_notional / entry_price

        adjusted_notional = adjusted_size * entry_price
        if adjusted_notional < MIN_NOTIONAL:
            return 0.0
        return round(adjusted_size, 3)

    def _safe_place_entry_order(self, params: dict):
        """Places entry order with 1-time timeout retry checking newClientOrderId"""
        client_id = params['newClientOrderId']
        
        try:
            return self._request("POST", "/fapi/v1/order", params, signed=True)
        except Timeout:
            logger.warning(f"Timeout placing entry order. Checking if {client_id} resolved...")
            try:
                # Query order status using client ID
                order_status = self._request("GET", "/fapi/v1/order", {
                    "symbol": self.symbol,
                    "origClientOrderId": client_id
                }, signed=True)
                
                logger.info(f"Order found on query: {order_status.get('status')} - DO NOT RETRY.")
                return order_status
            except ExecutorError as e:
                if "Code -2013" in str(e) or "doesn't exist" in str(e).lower():
                    # Order does NOT exist. We can safely retry once
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
        side = "BUY" if direction.upper() == "BUY" else "SELL"
        close_side = "SELL" if side == "BUY" else "BUY"

        if self.paper_mode:
            logger.info(f"[PAPER] {direction} {position_size:.6f} {settings.SYMBOL} entry~{entry_price:,.2f}")
            return {
                "id": f"paper_{int(time.time())}",
                "status": "open",
                "paper": True,
                "direction": direction,
                "symbol": settings.SYMBOL,
                "size": position_size,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "signal_ts": signal_ts,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "exchange": "binance",
                "order_type": settings.ORDER_TYPE,
                "strategy_type": strategy_type,
            }

        if not self.connected:
            raise ExecutorError("Not connected.")

        position_size = self._validate_margin(position_size, entry_price)
        if position_size <= 0:
            raise ExecutorError("Position size too small after margin validation.")

        # \u2500\u2500 1. Entry Order \u2500\u2500
        order_type = settings.ORDER_TYPE.upper()
        # Strategy metadata in clientOrderId for explainability
        strat_prefix = strategy_type[:5] if strategy_type else "legcy"
        unique_client_id = f"tradia_{strat_prefix}_{uuid.uuid4().hex[:10]}"
        
        params = {
            "symbol": self.symbol,
            "side": side,
            "type": order_type,
            "quantity": round(position_size, 3),
            "positionSide": "BOTH",
            "newClientOrderId": unique_client_id
        }
        
        if order_type == "LIMIT":
            params["timeInForce"] = "GTC"
            offset = 1.0001 if side == "BUY" else 0.9999
            params["price"] = round(entry_price * offset, 2)
        
        try:
            entry_res = self._safe_place_entry_order(params)
            logger.info(f"Entry {order_type} resolved: {entry_res.get('orderId')} status={entry_res.get('status')}")
        except Exception as e:
            raise ExecutorError(f"Entry Order failed completely: {e}")

        # \u2500\u2500 2. Determine SL/TP Size (Partial Fills Handling) \u2500\u2500
        # If it's a Limit order and fills partially, we only place SL/TP for executed amount
        # If executedQty is zero (just placed in book), we fall back to requested size, 
        # or we just use Binance closePosition=True to avoid size explicitly!
        
        try:
            executed_qty = float(entry_res.get("executedQty", 0))
            if executed_qty > 0:
                sltp_size = executed_qty
                logger.info(f"Using executedQty: {sltp_size} for conditional orders.")
            else:
                sltp_size = round(position_size, 3)
                logger.info(f"Order resting or no executedQty. Using requested size: {sltp_size}")

            # 3. Stop Loss
            self._request("POST", "/fapi/v1/order", {
                "symbol": self.symbol,
                "side": close_side,
                "type": "STOP_MARKET",
                "quantity": sltp_size,
                "stopPrice": round(sl_price, 2),
                "reduceOnly": "true",
                "workingType": "MARK_PRICE"
            }, signed=True)
            logger.info(f"SL placed at {sl_price}")

            # 4. Take Profit
            self._request("POST", "/fapi/v1/order", {
                "symbol": self.symbol,
                "side": close_side,
                "type": "TAKE_PROFIT_MARKET",
                "quantity": sltp_size,
                "stopPrice": round(tp_price, 2),
                "reduceOnly": "true",
                "workingType": "MARK_PRICE"
            }, signed=True)
            logger.info(f"TP placed at {tp_price}")

            return {
                "id": str(entry_res.get("orderId")),
                "status": "open",
                "paper": False,
                "direction": direction,
                "symbol": settings.SYMBOL,
                "size": float(sltp_size),
                "entry_price": float(entry_res.get("avgPrice") or entry_res.get("price") or entry_price),
                "sl_price": sl_price,
                "tp_price": tp_price,
                "signal_ts": signal_ts,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "exchange": "binance",
                "order_type": settings.ORDER_TYPE,
                "strategy_type": strategy_type,
            }

        except Exception as e:
            logger.error(f"Failed to place SL/TP conditionally! {e}")
            # Emergency fallback: we have a position but SL hit an error. We should leave it open for now
            # but log critically or attempt to close depending on risk model. We'll raise it so live_trader logs it.
            raise ExecutorError(f"SL/TP Placement failed: {e}")

    def close_position(self, direction: str, size: float, reason: str = "manual") -> Optional[dict]:
        if self.paper_mode:
            return {"status": "closed", "paper": True, "reason": reason}

        if not self.connected:
            raise ExecutorError("Not connected.")

        close_side = "SELL" if direction.upper() == "BUY" else "BUY"
        
        try:
            self.cancel_all_conditional_orders()
            res = self._request("POST", "/fapi/v1/order", {
                "symbol": self.symbol,
                "side": close_side,
                "type": "MARKET",
                "quantity": round(size, 3),
                "reduceOnly": "true"
            }, signed=True)
            return {
                "status": "closed", 
                "paper": False, 
                "order_id": str(res.get("orderId")), 
                "reason": reason
            }
        except Exception as e:
            raise ExecutorError(f"Close failed: {e}")

    def cancel_all_conditional_orders(self, symbol: Optional[str] = None) -> None:
        """
        Only cancels STOP_MARKET and TAKE_PROFIT_MARKET orders.
        Leaves rest limit/entry orders untouched.
        """
        if self.paper_mode or not self.connected:
            return
        sym = symbol.replace("/", "").replace(":USDT", "") if symbol else self.symbol
        try:
            # Get open orders
            orders = self._request("GET", "/fapi/v1/openOrders", {"symbol": sym}, signed=True)
            
            cancel_count = 0
            for order in orders:
                if order.get("type") in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
                    self._request("DELETE", "/fapi/v1/order", {
                        "symbol": sym,
                        "orderId": order["orderId"]
                    }, signed=True)
                    cancel_count += 1
                    
            logger.info(f"Cancelled {cancel_count} conditional SL/TP orders.")
        except Exception as e:
            logger.warning(f"Failed to cleanly cancel conditional orders: {e}")

    def get_closed_pnl(self, limit: int = 50) -> list[dict]:
        if not self.connected:
            return []
        try:
            res = self._request("GET", "/fapi/v1/userTrades", {"symbol": self.symbol, "limit": limit}, signed=True)
            result = []
            for r in res:
                result.append({
                    "order_id": r.get("orderId"),
                    "symbol": r.get("symbol"),
                    "side": str(r.get("side")).lower(),
                    "price": float(r.get("price", 0)),
                    "amount": float(r.get("qty", 0)),
                    "pnl": float(r.get("realizedPnl", 0)),
                    "fee": float(r.get("commission", 0)),
                    "timestamp": r.get("time"),
                    "datetime": datetime.fromtimestamp(r.get("time", 0) / 1000, tz=timezone.utc).isoformat()
                })
            return result
        except Exception as e:
            logger.error(f"get_closed_pnl failed: {e}")
            return []
TradeExecutor = BinanceFuturesExecutor

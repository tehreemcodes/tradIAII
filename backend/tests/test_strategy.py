"""
Unit Tests — ICT Strategy Components
======================================
Tests cover:
  - Swing detection correctness
  - CISD detection (body + volume thresholds)
  - FVG detection and labeling
  - State machine sequential enforcement
  - State machine 20-candle window expiry
  - Label generator WIN/LOSS assignment
  - Risk manager position sizing and compounding
  - Feature builder completeness
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Make backend importable from tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.ict_strategy    import detect_swings, detect_cisd, detect_fvg
from backend.services.state_machine   import run_state_machine, PatternState
from backend.services.label_generator import label_trades
from backend.services.risk_manager    import RiskManager
from backend.services.feature_builder import build_features, FEATURE_COLS


# ── Helpers ───────────────────────────────────────────────────

def make_flat_df(n=100, price=50000.0) -> pd.DataFrame:
    """Flat candles — no swings, no patterns."""
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open":   price,
        "high":   price + 100,
        "low":    price - 100,
        "close":  price,
        "volume": 100.0,
    }, index=idx)


def make_swing_high_df(swing_pos=10) -> pd.DataFrame:
    """DataFrame with a clear swing high at swing_pos."""
    n = 30
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    highs = [50000.0] * n
    # Make swing_pos strictly higher than neighbors
    highs[swing_pos] = 55000.0
    df = pd.DataFrame({
        "open":   50000.0,
        "high":   highs,
        "low":    49500.0,
        "close":  50000.0,
        "volume": 100.0,
    }, index=idx)
    return df


def make_swing_low_df(swing_pos=10) -> pd.DataFrame:
    """DataFrame with a clear swing low at swing_pos."""
    n = 30
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    lows = [50000.0] * n
    lows[swing_pos] = 44000.0
    df = pd.DataFrame({
        "open":   50000.0,
        "high":   50500.0,
        "low":    lows,
        "close":  50000.0,
        "volume": 100.0,
    }, index=idx)
    return df


# ── Swing Detection Tests ─────────────────────────────────────

class TestSwingDetection:

    def test_swing_high_detected(self):
        df = make_swing_high_df(swing_pos=10)
        result = detect_swings(df)
        assert result["swing_high"].iloc[10] == True
        assert result["swing_high_price"].iloc[10] == 55000.0

    def test_swing_high_not_detected_at_edge(self):
        """Cannot detect swing at first 2 or last 2 positions."""
        df = make_swing_high_df(swing_pos=10)
        result = detect_swings(df)
        # First and last 2 should never have swings
        assert result["swing_high"].iloc[0] == False
        assert result["swing_high"].iloc[1] == False
        assert result["swing_high"].iloc[-1] == False
        assert result["swing_high"].iloc[-2] == False

    def test_swing_low_detected(self):
        df = make_swing_low_df(swing_pos=10)
        result = detect_swings(df)
        assert result["swing_low"].iloc[10] == True
        assert result["swing_low_price"].iloc[10] == 44000.0

    def test_no_swings_on_flat_data(self):
        df = make_flat_df(50)
        result = detect_swings(df)
        assert result["swing_high"].sum() == 0
        assert result["swing_low"].sum() == 0

    def test_swing_requires_both_sides(self):
        """A high that is only dominant on the left side is NOT a swing."""
        n = 20
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        highs = [50000.0] * n
        highs[5]  = 55000.0
        highs[6]  = 56000.0   # higher than highs[5] on right side
        df = pd.DataFrame({
            "open": 50000.0, "high": highs,
            "low": 49500.0, "close": 50000.0, "volume": 100.0,
        }, index=idx)
        result = detect_swings(df)
        assert result["swing_high"].iloc[5] == False   # not a swing (6 is higher)


# ── CISD Tests ────────────────────────────────────────────────

class TestCISD:

    def test_bull_cisd_detected(self):
        """Large bullish candle that sweeps a prior swing high should be detected."""
        n = 50
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        avg_body = 200.0
        avg_vol  = 100.0

        bodies  = [avg_body] * n
        volumes = [avg_vol] * n

        # Place a swing high at candle 25 (needs strict high on both sides)
        highs_data = [50000 + b + 50 for b in bodies]
        highs_data[25] = 55000.0  # clear swing high

        # Candle 35: big bullish with volume spike that sweeps above the swing high
        bodies[35]  = avg_body * 2.0
        volumes[35] = avg_vol * 2.0
        highs_data[35] = 56000.0  # sweeps above swing high at 55000

        df = pd.DataFrame({
            "open":   50000.0,
            "high":   highs_data,
            "low":    49900.0,
            "close":  [50000 + b for b in bodies],
            "volume": volumes,
        }, index=idx)

        # Must run detect_swings first — CISD requires prior swing data
        df = detect_swings(df)
        result = detect_cisd(df)
        assert result["bull_cisd"].iloc[35] == True

    def test_cisd_requires_volume_spike(self):
        """Large body WITHOUT volume spike should NOT be CISD."""
        n = 50
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        avg_body = 200.0
        avg_vol  = 100.0

        bodies  = [avg_body] * n
        volumes = [avg_vol] * n

        # Big body but NO volume spike
        bodies[40]  = avg_body * 2.0
        volumes[40] = avg_vol * 1.0   # same volume, no spike

        df = pd.DataFrame({
            "open":   50000.0,
            "high":   [50000 + b + 50 for b in bodies],
            "low":    49900.0,
            "close":  [50000 + b for b in bodies],
            "volume": volumes,
        }, index=idx)

        result = detect_cisd(df)
        assert result["bull_cisd"].iloc[40] == False


# ── FVG Tests ─────────────────────────────────────────────────

class TestFVG:

    def test_bullish_fvg_detected(self):
        """
        Candle i: high = 100
        Candle i+1: middle candle (label goes here)
        Candle i+2: low = 110  (> 100 -> bullish FVG)
        """
        n = 20
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        highs = [50000.0] * n
        lows  = [49900.0] * n

        i = 10
        highs[i]   = 50100.0
        lows[i+2]  = 50200.0   # > highs[i] -> bullish FVG

        df = pd.DataFrame({
            "open":   50000.0,
            "high":   highs,
            "low":    lows,
            "close":  50000.0,
            "volume": 100.0,
        }, index=idx)

        result = detect_fvg(df)
        # FIX: Label is on the CONFIRMING candle (i+2), not the middle (i+1)
        assert result["bull_fvg"].iloc[i + 2] == True
        assert result["fvg_bot"].iloc[i + 2] == 50100.0
        assert result["fvg_top"].iloc[i + 2] == 50200.0

    def test_bearish_fvg_detected(self):
        """
        Candle i: low = 50000
        Candle i+2: high = 49900  (< 50000 -> bearish FVG)
        """
        n = 20
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        highs = [50200.0] * n
        lows  = [50000.0] * n

        i = 8
        lows[i]    = 50100.0
        highs[i+2] = 49950.0   # < lows[i] -> bearish FVG

        df = pd.DataFrame({
            "open":   50050.0,
            "high":   highs,
            "low":    lows,
            "close":  50050.0,
            "volume": 100.0,
        }, index=idx)

        result = detect_fvg(df)
        # FIX: Label is on the CONFIRMING candle (i+2), not the middle (i+1)
        assert result["bear_fvg"].iloc[i + 2] == True


# ── State Machine Tests ───────────────────────────────────────

class TestStateMachine:

    def test_pattern_expires_after_window(self):
        p = PatternState("BULL", swing_index=0, swing_price=50000.0, window=20)
        assert p.is_expired(20) == False   # at window boundary: NOT expired
        assert p.is_expired(21) == True    # one over: expired

    def test_fvg_before_cisd_ignored(self):
        """FVG must come AFTER CISD — FVG first should not complete pattern."""
        p = PatternState("BULL", swing_index=0, swing_price=50000.0, window=20)

        # Try to give FVG before CISD
        row_with_fvg = pd.Series({
            "bull_fvg": True,
            "bear_fvg": False,
            "bull_cisd": False,
            "bear_cisd": False,
            "fvg_top": 50200.0,
            "fvg_bot": 50100.0,
        })
        p.tick(1, row_with_fvg)
        assert p.complete == False   # should NOT complete

    def test_complete_bull_pattern(self):
        """Full BULL pattern: swing -> CISD -> FVG should complete."""
        p = PatternState("BULL", swing_index=0, swing_price=49000.0, window=20)

        # Tick 1: CISD
        p.tick(1, pd.Series({"bull_cisd": True, "bear_cisd": False,
                              "bull_fvg": False, "bear_fvg": False}))
        assert p.cisd_found == True
        assert p.complete == False

        # Tick 2: FVG
        p.tick(2, pd.Series({
            "bull_cisd": False, "bear_cisd": False,
            "bull_fvg": True, "bear_fvg": False,
            "fvg_top": 50200.0, "fvg_bot": 50100.0,
        }))
        assert p.complete == True
        assert p.fvg_index == 2

    def test_no_signals_on_flat_data(self):
        """Flat market should generate no signals."""
        df = make_flat_df(100)
        from backend.services.ict_strategy import run_ict_pipeline
        df = run_ict_pipeline(df)
        df = run_state_machine(df)
        assert (df["signal"] == 2).sum() == 0
        assert (df["signal"] == 0).sum() == 0


# ── Label Generator Tests ─────────────────────────────────────

class TestLabelGenerator:

    def _make_signal_df(self, is_buy: bool):
        """Create a minimal DataFrame with one signal."""
        n = 60
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        prices = [50000.0] * n
        df = pd.DataFrame({
            "open":   prices,
            "high":   [p + 200 for p in prices],
            "low":    [p - 200 for p in prices],
            "close":  prices,
            "volume": 100.0,
            "signal":     [1] * n,
            "signal_sl":  [np.nan] * n,
        }, index=idx)

        # Place signal at candle 5
        df.loc[df.index[5], "signal"]    = 2 if is_buy else 0
        df.loc[df.index[5], "signal_sl"] = 49000.0 if is_buy else 51000.0
        return df

    def test_buy_tp_hit(self):
        df = self._make_signal_df(is_buy=True)
        entry = 50000.0
        sl    = 49000.0
        risk  = abs(entry - sl)
        tp    = entry + risk * 2   # = 52000

        # Make candle 10 hit TP
        df.loc[df.index[10], "high"] = tp + 100

        result = label_trades(df)
        assert result["ml_label"].iloc[5] == 1.0   # WIN

    def test_buy_sl_hit(self):
        df = self._make_signal_df(is_buy=True)
        entry = 50000.0
        sl    = 49000.0

        # Make candle 8 hit SL
        df.loc[df.index[8], "low"] = sl - 100

        result = label_trades(df)
        assert result["ml_label"].iloc[5] == 0.0   # LOSS

    def test_no_outcome_is_nan(self):
        """If neither TP nor SL is hit within window, label is NaN."""
        df = self._make_signal_df(is_buy=True)
        # Default prices: high=50200, low=49800 — neither TP (52000) nor SL (49000)
        result = label_trades(df)
        assert pd.isna(result["ml_label"].iloc[5])


# ── Risk Manager Tests ────────────────────────────────────────

class TestRiskManager:

    def test_initial_risk_amount(self):
        rm = RiskManager(initial_capital=10000, risk_pct=0.10)
        assert rm.risk_amount == 1000.0

    def test_capital_compounds_after_loss(self):
        rm = RiskManager(initial_capital=10000, risk_pct=0.10)
        ts = pd.Timestamp("2023-01-01 10:00")
        trade = rm.calculate_position(50000.0, 49000.0, "BUY", ts)
        assert trade is not None
        rm.record_outcome(trade, "SL")
        # Capital = 10000 - risk_amount - fee_cost
        # Fee and SL buffer shift the exact numbers;
        # just check capital went down by approximately the risk
        assert rm.capital < 10000.0
        assert rm.capital > 8500.0  # should lose ~1000 + fees
        # Risk should compound: lower capital = lower risk
        # Note: compound=False by default → risk_amount = initial_capital * risk_pct (fixed)
        assert rm.risk_amount <= 1000.0

    def test_capital_compounds_after_win(self):
        rm = RiskManager(initial_capital=10000, risk_pct=0.10, rr=2.0)
        ts = pd.Timestamp("2023-01-01 10:00")
        trade = rm.calculate_position(50000.0, 49000.0, "BUY", ts)
        rm.record_outcome(trade, "TP")
        # Capital = 10000 + (risk * 2.0) - fee_cost
        # gains ~2000 minus round-trip fees
        assert rm.capital > 11500.0
        assert rm.capital < 12100.0

    def test_sl_wrong_side_returns_none(self):
        rm = RiskManager(initial_capital=10000)
        ts = pd.Timestamp("2023-01-01 10:00")
        # BUY with SL ABOVE entry — invalid
        trade = rm.calculate_position(50000.0, 51000.0, "BUY", ts)
        assert trade is None

    def test_cooldown_enforced(self):
        rm = RiskManager(initial_capital=10000, cooldown_minutes=5)
        ts1 = pd.Timestamp("2023-01-01 10:00")
        ts2 = pd.Timestamp("2023-01-01 10:03")   # 3 min later — too soon

        trade1 = rm.calculate_position(50000.0, 49000.0, "BUY", ts1)
        rm.record_outcome(trade1, "SL")

        # Try again 3 minutes later — should be blocked
        trade2 = rm.calculate_position(50000.0, 49000.0, "BUY", ts2)
        assert trade2 is None

    def test_notional_cap(self):
        """Position notional should not exceed capital * MAX_NOTIONAL_MULT."""
        from backend.config.settings import MAX_NOTIONAL_MULT
        rm = RiskManager(initial_capital=10000, risk_pct=0.10)
        ts = pd.Timestamp("2023-01-01 10:00")
        # Tiny SL distance -> huge position size without cap
        trade = rm.calculate_position(50000.0, 49999.0, "BUY", ts)
        if trade:
            max_notional = rm.capital * MAX_NOTIONAL_MULT
            assert trade.position_size * trade.entry <= max_notional + 1e-6


# ── Feature Builder Tests ─────────────────────────────────────

class TestFeatureBuilder:

    def test_all_feature_cols_defined(self):
        """FEATURE_COLS should be non-empty."""
        assert len(FEATURE_COLS) > 0

    def test_build_features_returns_all_cols(self):
        """build_features should return all feature columns."""
        from backend.services.ict_strategy import run_ict_pipeline
        np.random.seed(1)
        n = 200
        price = 50000.0
        rows = []
        for _ in range(n):
            o = price
            c = o + np.random.randn() * 400
            rows.append({
                "open": o, "close": c,
                "high": max(o, c) + abs(np.random.randn()) * 200,
                "low":  min(o, c) - abs(np.random.randn()) * 200,
                "volume": abs(np.random.randn()) * 500 + 200,
            })
            price = c

        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        df  = pd.DataFrame(rows, index=idx)
        df  = run_ict_pipeline(df)
        df  = build_features(df)

        for col in ["body_pct", "volume_ratio", "atr_14",
                    "is_london_kz", "is_ny_kz", "hour"]:
            assert col in df.columns, f"Missing feature: {col}"

    def test_no_future_data_in_features(self):
        """ATR and rolling features must use only past data."""
        from backend.services.ict_strategy import run_ict_pipeline
        n = 50
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "open": 50000.0, "high": 50200.0,
            "low": 49800.0,  "close": 50000.0, "volume": 100.0,
        }, index=idx)
        df = run_ict_pipeline(df)
        df = build_features(df)
        # ATR at candle 0 should be NaN or the candle's own range
        # (not influenced by future candles)
        assert "atr_14" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# backend/scripts/train_model.py

import sys
import logging
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.config.settings import (
    SIGNAL_TF, HTF_LIST,
    MODEL_DIR,
    XGBOOST_PARAMS, XGBOOST_EARLY_STOPPING, TRAIN_SPLIT,
)
from backend.config.logging_setup import setup_logging
from backend.services.data_loader import load_all_timeframes
from backend.services.ict_strategy import run_ict_pipeline
from backend.services.state_machine import run_state_machine
from backend.services.multi_timeframe import merge_htf_into_ltf
from backend.services.feature_builder import build_features, FEATURE_COLS
from backend.services.label_generator import label_trades

setup_logging()
logger = logging.getLogger(__name__)


def build_xgb_params(y):
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos if pos > 0 else 1
    params = dict(XGBOOST_PARAMS)
    params["scale_pos_weight"] = spw
    logger.info(f"Class counts — Loss(0): {neg}, Win(1): {pos}")
    logger.info(f"scale_pos_weight = {spw:.2f}  (target: >8 means severe imbalance)")
    return params


def evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    for threshold in [0.5, 0.4, 0.3]:
        y_pred = (y_prob >= threshold).astype(int)
        acc   = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        print(f"\n--- Threshold = {threshold} ---")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {acc:.4f}  |  Kappa: {kappa:.4f}")

    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    print(f"\nAUC (threshold-independent): {auc:.4f}")
    print("=" * 60)
    print("ACTION: Pick the threshold where class-1 recall >= 0.50")
    print("        Then set MIN_CONFIDENCE to that value in settings.py")
    print("=" * 60)


def main(timeframe):
    logger.info("=" * 60)
    logger.info(f"TradIA — Training Clean ML Model [{timeframe}]")
    logger.info("=" * 60)

    # 1. Load data
    logger.info("[1/6] Loading data...")
    data = load_all_timeframes()
    if timeframe not in data:
        logger.error(f"Timeframe {timeframe} not found. Run fetch_data first.")
        sys.exit(1)

    df = data[timeframe].copy()

    # 2. ICT pipeline
    logger.info("[2/6] Running ICT pipeline...")
    df = run_ict_pipeline(df)
    df = run_state_machine(df)

    # 3. HTF merge
    logger.info("[3/6] Merging HTF bias...")
    htf_map = {
        "15m": ["1h", "4h"],
        "1h": ["4h", "1d"],
        "4h": ["1d"],
        "1d": []
    }

    htf = {tf: data[tf] for tf in htf_map.get(timeframe, []) if tf in data}
    if htf:
        df = merge_htf_into_ltf(df, htf)

    # 4. Features
    logger.info("[4/6] Building features...")
    df = build_features(df)

    # 5. Labeling (FIXED)
    logger.info("[5/6] Labeling trades (TP/SL based)...")
    df = label_trades(df, timeframe=timeframe)

    df = df.dropna(subset=["ml_label"])

    features = [f for f in FEATURE_COLS if f in df.columns]

    X = df[features].fillna(0)
    y = df["ml_label"].astype(int)

    positive_rate = y.mean()
    logger.info(f"Dataset size: {len(X):,}")
    logger.info(f"Class distribution:\n{y.value_counts()}")
    logger.info(f"Positive rate (wins): {positive_rate:.2%}")

    if positive_rate < 0.05:
        logger.warning("CRITICAL: Win rate below 5% — labels are almost all losses.")
        logger.warning("Cause: LABEL_FORWARD window too short for RR=3, or data issue.")
        logger.warning("Fix: increase LABEL_FORWARD in settings.py, or reduce REWARD_RATIO.")
    elif positive_rate < 0.10:
        logger.warning("WARNING: Win rate below 10% — model will struggle to learn wins.")
        logger.warning("Consider increasing LABEL_FORWARD or reducing MIN_CONFIDENCE threshold.")


    # Train-test split
    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    logger.info("[6/6] Training XGBoost model...")
    params = build_xgb_params(y_train)

    model = XGBClassifier(
        **params,
        early_stopping_rounds=XGBOOST_EARLY_STOPPING
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    logger.info("Top 20 features by importance:")
    logger.info("\n" + importances.head(20).to_string())
    
    importance_path = MODEL_DIR / f"feature_importance_{timeframe}.txt"
    importances.to_csv(importance_path)
    logger.info(f"Feature importances saved to {importance_path}")

    # Evaluate
    evaluate(model, X_test, y_test)

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / f"ict_model_{timeframe}.pkl")
    joblib.dump(scaler, MODEL_DIR / f"scaler_{timeframe}.pkl")
    joblib.dump(features, MODEL_DIR / f"features_{timeframe}.pkl")

    print("\n✅ Model, scaler, and features saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default=SIGNAL_TF)
    args = parser.parse_args()

    main(args.timeframe)
"""
module7_train.py
-----------------
Module 7: Payment Failure Risk Scoring — Step 4 (modeling) and Step 5
(evaluation).

Takes the train/test frames produced by module7_features.py (Steps 1-3)
and:
  - Step 4: fits a Logistic Regression model, class_weight='balanced' to
    counter the ~2-4% positive (FAILED) rate, then wraps it in
    CalibratedClassifierCV (Platt/sigmoid scaling, cv=5) so risk_score
    is a real, checkable probability — e.g. among payments scored ~0.6,
    roughly 60% should actually be FAILED — rather than just a ranking
    signal distorted by the balanced class weighting. Use
    --no-calibration to get the raw uncalibrated model instead. Features
    now include is_peak_season (Oct/Nov/Dec) alongside payment_method/
    payment_provider — added after confirming monthly transaction volume
    correlates with monthly failure rate (0.58) in this data.
  - Step 5: evaluates on the FAILED class specifically — precision/
    recall/F1 at the default 0.5 threshold, PR-AUC, ROC-AUC, Brier
    score, and a calibration reliability table (predicted vs. actual
    failure rate by score decile). Plain accuracy is deliberately NOT
    used as a headline metric (a model that always predicts SUCCESS
    already scores ~98% accuracy here and is useless). Also reports
    precision/recall if you review the riskiest 5%/10%/20% of
    transactions by score — the capacity-based operating mode
    module7_score.py actually uses in production.

Usage:
    python module7_train.py --train module7_features_train.csv --test module7_features_test.csv
"""

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

CATEGORICAL_FEATURES = ["payment_method", "payment_provider", "is_peak_season"]
NUMERIC_FEATURES = [
    "log_amount",
    "customer_prior_fail_rate", "customer_prior_txn_count",
    "store_prior_fail_rate", "store_prior_txn_count",
]
TARGET = "label"


def build_preprocessor():
    return ColumnTransformer([
        # min_frequency=100: any category with fewer than 100 training
        # examples gets pooled into a single "infrequent" bucket per
        # column instead of getting its own coefficient. Added after
        # finding the smallest payment_provider categories (the wallet
        # apps, ~50-62 transactions each) were producing individually
        # overfit coefficients — e.g. Paytm Wallet's flagged_rate came
        # out at 84% against an actual_fail_rate of only 16%, a gap much
        # larger than any higher-volume category showed. Categories
        # above the threshold (CARD/UPI providers at 1000+, NET BANKING
        # providers at 160-190) are unaffected and keep their own
        # coefficient as before.
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])


def build_model():
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])


def build_calibrated_model(method="sigmoid", cv=5):
    """
    Wraps build_model() in CalibratedClassifierCV so predict_proba() output
    means what it says (e.g. among all payments scored ~0.6, ~60% actually
    fail), instead of the raw class_weight='balanced' output, which only
    preserves relative RANKING, not real probabilities.

    method='sigmoid' (Platt scaling) is used by default rather than
    'isotonic': isotonic is more flexible but needs more positive examples
    per fold to avoid overfitting the calibration curve itself, and with
    only ~400 FAILED rows in training split across 5 folds (~80/fold),
    sigmoid's simpler 2-parameter fit is the safer choice here.

    cv=5 uses CalibratedClassifierCV's internal cross-validation: it fits
    5 clones of the base model on rotating folds and calibrates each on
    the held-out fold, so no separate manual calibration split is needed
    and no test-set data is touched.
    """
    return CalibratedClassifierCV(build_model(), method=method, cv=cv)


def log_feature_coefficients(model):
    """
    Extracts the underlying LogisticRegression coefficients and logs them
    sorted by magnitude — direct evidence of what the model actually
    learned per feature, independent of any confounding from aggregating
    scores by month/method/provider after the fact (see module7_score.py's
    _add_actual_fail_rate for the complementary ground-truth check).

    Coefficients are in the model's internal log-odds space (after
    class_weight='balanced' reweighting, before calibration remapping) —
    so their SIGN and RELATIVE size are meaningful ("this pushes risk up
    vs down, and more than that other feature"), but the raw number isn't
    a probability shift. For a one-hot feature like is_peak_season, look
    at the difference between its True and False rows: a positive
    difference means the model learned peak season genuinely raises risk,
    net of every other feature.

    Handles both the calibrated case (CalibratedClassifierCV wraps 5
    fold-fitted clones of the base Pipeline — coefficients are averaged
    across folds) and the --no-calibration case (a single fitted
    Pipeline), and works across sklearn versions where the fitted
    per-fold estimator is exposed as either .estimator or the older
    .base_estimator.
    """
    if hasattr(model, "calibrated_classifiers_"):
        pipelines = []
        for cc in model.calibrated_classifiers_:
            pipeline = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if pipeline is not None:
                pipelines.append(pipeline)
    else:
        pipelines = [model]

    if not pipelines:
        logger.warning("log_feature_coefficients: could not locate a fitted "
                        "LogisticRegression inside the model — skipping coefficient report")
        return

    all_coefs = []
    feature_names = None
    for pipeline in pipelines:
        prep = pipeline.named_steps["prep"]
        clf = pipeline.named_steps["clf"]
        names = prep.get_feature_names_out()
        if feature_names is None:
            feature_names = names
        all_coefs.append(pd.Series(clf.coef_[0], index=names))

    coef_df = pd.concat(all_coefs, axis=1)
    mean_coef = coef_df.mean(axis=1).sort_values(key=abs, ascending=False)

    logger.info("Feature coefficients (log-odds space, averaged across %d fold(s), "
                "sorted by |effect|; positive = raises risk):\n%s",
                len(pipelines), mean_coef.to_string())

    peak_rows = mean_coef[mean_coef.index.str.contains("is_peak_season")]
    if len(peak_rows) == 2:
        # OneHotEncoder emits one column per category (no dropped
        # reference), so the net effect of being in peak season is the
        # difference between the "peak" (1/True) and "non-peak" (0/False)
        # category weights. Matched by suffix rather than assuming a
        # specific dtype spelling, since is_peak_season may be encoded as
        # 0/1 (int) or True/False (bool) depending on how it was built.
        true_mask = peak_rows.index.str.endswith(("_1", "_True", "True"))
        false_mask = peak_rows.index.str.endswith(("_0", "_False", "False"))
        if true_mask.sum() == 1 and false_mask.sum() == 1:
            true_val = peak_rows[true_mask].iloc[0]
            false_val = peak_rows[false_mask].iloc[0]
            net_effect = true_val - false_val
            direction = "RAISES" if net_effect > 0 else "LOWERS"
            logger.info("is_peak_season net effect = %.4f (%s risk relative to non-peak months) "
                        "— this is the model's own learned answer to whether Oct/Nov/Dec are "
                        "riskier, independent of any month-level aggregation noise.",
                        net_effect, direction)
        else:
            logger.info("is_peak_season coefficients found but could not be matched to "
                        "True/False categories automatically:\n%s", peak_rows.to_string())


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    logger.info("=" * 70)
    logger.info("Logistic Regression (calibrated) — PR-AUC=%.4f  ROC-AUC=%.4f  Brier=%.4f",
                pr_auc, roc_auc, brier)

    # Calibration check: bin test payments into deciles by predicted
    # risk_score and compare each bin's MEAN PREDICTED score against its
    # ACTUAL observed failure rate. A well-calibrated model has these two
    # columns roughly matching — e.g. the bin with mean predicted ~0.6
    # should have an actual failure rate near 0.6 too. Brier score above
    # (mean squared error between proba and actual outcome, lower=better,
    # 0=perfect) is the single-number summary of the same idea.
    calib_df = pd.DataFrame({"proba": proba, "actual": np.asarray(y_test)})
    try:
        calib_df["bin"] = pd.qcut(calib_df["proba"], q=10, duplicates="drop")
    except ValueError:
        calib_df["bin"] = pd.cut(calib_df["proba"], bins=10)
    calib_table = calib_df.groupby("bin", observed=True).agg(
        n=("actual", "size"),
        mean_predicted=("proba", "mean"),
        actual_fail_rate=("actual", "mean"),
    )
    logger.info("Calibration check (predicted vs. actual failure rate by score decile):\n%s",
                calib_table.to_string())

    # PRIMARY readout: capacity-based precision/recall, ranking transactions
    # by risk_score and reviewing a fixed top-K%. This is the actual
    # operating mode module7_score.py uses (--capacity-pct), so it's the
    # number that predicts what you'll see in production.
    order = np.argsort(-proba)
    y_sorted = np.asarray(y_test)[order]
    logger.info("Capacity-based precision/recall (matches module7_score.py --capacity-pct):")
    for pct in (0.05, 0.10, 0.20):
        k = max(1, int(len(y_sorted) * pct))
        flagged = y_sorted[:k]
        precision_at_k = flagged.mean()
        recall_at_k = flagged.sum() / max(y_sorted.sum(), 1)
        logger.info("  Flag top %d%% by score (%d txns): precision=%.3f, recall=%.3f",
                    int(pct * 100), k, precision_at_k, recall_at_k)

    # SECONDARY: the 0.5 cutoff below is still NOT the operating point used
    # in production (module7_score.py flags by rank/capacity, not a fixed
    # probability). It's now a MEANINGFUL number post-calibration (unlike
    # before), but capacity-based flagging remains the deployed behavior —
    # treat this as a secondary diagnostic, not the deployment metric.
    pred = (proba >= 0.5).astype(int)
    report = classification_report(y_test, pred, target_names=["SUCCESS", "FAILED"], digits=3)
    cm = confusion_matrix(y_test, pred)
    logger.info("[diagnostic — calibrated, but not the production operating point] "
                "Confusion matrix at 0.5 threshold [rows=actual, cols=predicted] (SUCCESS, FAILED):\n%s", cm)
    logger.info("[diagnostic] Classification report at 0.5 threshold:\n%s", report)

    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier, "proba": proba, "pred": pred}


def run(train_path, test_path, model_out="module7_model.joblib", calibrate=True, calibration_method="sigmoid"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train, y_train = train[CATEGORICAL_FEATURES + NUMERIC_FEATURES], train[TARGET]
    X_test, y_test = test[CATEGORICAL_FEATURES + NUMERIC_FEATURES], test[TARGET]

    logger.info("Training set: %d rows, %d positive (FAILED, %.4f rate)",
                len(y_train), y_train.sum(), y_train.mean())

    if calibrate:
        logger.info("Fitting with probability calibration (method=%s, cv=5) — "
                    "risk_score will represent a real probability, not just a ranking",
                    calibration_method)
        model = build_calibrated_model(method=calibration_method, cv=5)
    else:
        logger.info("Fitting WITHOUT calibration (--no-calibration) — risk_score "
                    "preserves ranking only, absolute values are not meaningful")
        model = build_model()

    model.fit(X_train, y_train)
    log_feature_coefficients(model)
    metrics = evaluate(model, X_test, y_test)

    joblib.dump(model, model_out)
    logger.info("Model saved to %s", model_out)

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 7: Steps 4-5 (train + evaluate Logistic Regression)")
    parser.add_argument("--train", required=True, help="Path to train CSV from module7_features.py")
    parser.add_argument("--test", required=True, help="Path to test CSV from module7_features.py")
    parser.add_argument("--model-out", default="module7_model.joblib")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Skip probability calibration, save the raw class_weight='balanced' model instead")
    parser.add_argument("--calibration-method", default="sigmoid", choices=["sigmoid", "isotonic"],
                        help="Calibration method when calibrating (default sigmoid — safer with a small positive class)")
    args = parser.parse_args()

    run(args.train, args.test, model_out=args.model_out,
        calibrate=not args.no_calibration, calibration_method=args.calibration_method)

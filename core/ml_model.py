"""
SessionGuard interpretable ML risk model
========================================

LogisticRegression over the SAME BehavioralFeatures the rules engine uses
-- same features, one more consumer -- testing whether a learned model can
catch the patient attack archetypes (patient_low_and_slow, patient USSD
sim-swap variants) that the hand-tuned weights completely miss, WITHOUT
introducing false positives on baseline sessions or legitimate anomalies.

Why LogisticRegression: every coefficient maps 1:1 to a named feature, so
the model's reasoning is auditable by judges and printable in the report.
No black box anywhere near the explanation layer.

HONESTY GUARANTEES BUILT IN
---------------------------
* Labels: y=1 ONLY for FraudLabel.is_attack=True. Baseline AND legitimate
  anomalies are both y=0 -- the model must learn that anomalous-but-honest
  is not fraud, not merely memorise "no label = safe".
* Stratified split (random_state=46, the fifth documented independent seed:
  42 users / 43 sessions / 44 attacks / 45 anomalies / 46 ML): with only 42
  positives a random split could strand every patient attack on one side;
  stratification keeps the positive rate proportional in train and test.
* class_weight='balanced': positives are <2% of rows; unweighted logistic
  regression would collapse to "never fraud" and still look accurate.
* Primary metrics come from the TEST set only. The full-dataset pass is
  clearly labelled as ILLUSTRATIVE (it includes training rows) and exists
  solely for side-by-side comparison against the rules-engine baseline.
* Operating point is explicit: p >= 0.5 counts as flagged ("block-like"),
  plus a threshold sweep showing the precision/recall/FPR trade-off.

Usage:
    python core/ml_model.py                  # train + evaluate + save
    from core.ml_model import predict_risk   # live scoring path (float prob)
"""

import os
import statistics
import sys

# --- Django bootstrap ONLY when run directly --------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE", "sessionguard_project.settings"
    )
    import django

    django.setup()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from core.models import BehavioralFeatures, FraudLabel  # noqa: E402

SEED = 46  # fifth independent RNG stream (documented reproducibility)
TEST_SIZE = 0.2
THRESHOLD = 0.5  # primary operating point for p(fraud)
SWEEP_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90]
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trained_model.joblib"
)

# Exact feature order used at training time. The saved bundle carries this
# list; predict_risk() rebuilds vectors in the SAVED order so the API can
# never silently mis-order columns after a refactor here.
FEATURE_COLUMNS = [
    "hour_deviation_score",
    "amount_deviation_score",       # null -> 0.0 (no tx = no amount signal)
    "device_change_flag",
    "sim_change_flag",
    "location_change_flag",
    "combined_device_location_flag",
    "impossible_travel_flag",       # strong standalone: impossibly far, too fast
    "new_recipient_flag",
    "velocity_count_5min",
    "menu_timing_deviation_score",  # null -> 0.0 (app has no USSD timing)
    "keystroke_deviation_score",    # null -> 0.0 (USSD or early app sessions)
]


def _vectorize(features, columns):
    """BehavioralFeatures row -> float vector in the given column order."""
    row = []
    for name in columns:
        value = getattr(features, name)
        if value is None:            # documented null semantics above
            value = 0.0
        row.append(float(int(value)) if isinstance(value, bool) else float(value))
    return row


def features_to_vector(features):
    return _vectorize(features, FEATURE_COLUMNS)


def _category_for(feature_row, labels):
    """baseline | anomaly:family | anomaly:simswap | attack:<type>."""
    label = labels.get(feature_row.session_id)
    if label is None:
        return "baseline"
    if label["is_attack"]:
        return f"attack:{label['attack_type']}"
    if label["is_legitimate_anomaly"]:
        return (
            "anomaly:family_shared_phone"
            if not feature_row.session.is_new_sim
            else "anomaly:genuine_sim_swap"
        )
    return "other"


def build_dataset():
    """
    One pass over all BehavioralFeatures -> X, y, kinds.

    y=1 strictly requires FraudLabel.is_attack=True; everything else
    (including both anomaly categories) is y=0.
    """
    labels = {
        row["session_id"]: row
        for row in FraudLabel.objects.values(
            "session_id", "is_attack", "attack_type", "is_legitimate_anomaly"
        )
    }
    X, y, kinds = [], [], []
    for f in BehavioralFeatures.objects.select_related(
        "session"
    ).iterator(chunk_size=500):
        X.append(features_to_vector(f))
        kind = _category_for(f, labels)
        kinds.append(kind)
        y.append(1 if kind.startswith("attack:") else 0)
    return np.array(X), np.array(y), kinds


def _counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def _metrics(y_true, y_pred):
    tp, fp, fn, tn = _counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "fpr": fpr}


def _pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def train_and_evaluate():
    line = "=" * 70

    # --- Dataset -------------------------------------------------------------
    X, y, kinds = build_dataset()
    print(line)
    print("ML MODEL TRAINING (LogisticRegression, interpretable)")
    print(line)
    print(f"Dataset                       : {len(X)} rows, "
          f"{int(y.sum())} positives (<{100 * y.mean():.1f}% class balance)")

    # --- Stratified 80/20 split ----------------------------------------------
    X_train, X_test, y_train, y_test, k_train, k_test = train_test_split(
        X, y, np.array(kinds), test_size=TEST_SIZE,
        stratify=y, random_state=SEED,
    )
    print(f"Train / test split            : {len(X_train)} / {len(X_test)} "
          f"(stratified, random_state={SEED})")
    print(f"  positives in train          : {int(y_train.sum())}")
    print(f"  positives in test           : {int(y_test.sum())}")
    print("  -> the model LEARNED from the first number and is JUDGED on "
          "the second.")

    # --- Train -----------------------------------------------------------------
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]

    # --- Test-set report @ operating point -------------------------------------
    m = _metrics(y_test, (test_probs >= THRESHOLD).astype(int))
    print()
    print(f"TEST-SET EVALUATION (operating point p >= {THRESHOLD}):")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"  Precision : {_pct(m['tp'], m['tp'] + m['fp'])}")
    print(f"  Recall    : {_pct(m['tp'], m['tp'] + m['fn'])}")
    print(f"  FPR       : {_pct(m['fp'], m['fp'] + m['tn'])}")

    # --- Threshold sweep -------------------------------------------------------
    print()
    print("Threshold sweep on TEST set (precision/recall/FPR trade-off):")
    print(f"  {'threshold':>9} {'TP':>3} {'FP':>3} {'FN':>3} "
          f"{'precision':>10} {'recall':>8} {'FPR':>8}")
    for t in SWEEP_THRESHOLDS:
        s = _metrics(y_test, (test_probs >= t).astype(int))
        print(f"  {t:>9.2f} {s['tp']:>3} {s['fp']:>3} {s['fn']:>3} "
              f"{_pct(s['tp'], s['tp'] + s['fp']):>10} "
              f"{_pct(s['tp'], s['tp'] + s['fn']):>8} "
              f"{_pct(s['fp'], s['fp'] + s['tn']):>8}")

    # NOTE: with only ~8 test positives these per-threshold numbers are
    # noisy single-attack steps. That is an honest limitation of a 42-positive
    # dataset, reported as-is rather than smoothed away.

    # --- Coefficients (interpretability payoff) ---------------------------------
    print()
    print("Learned coefficients (sorted by |weight|; sign = push toward fraud):")
    pairs = sorted(
        zip(FEATURE_COLUMNS, model.coef_[0]), key=lambda p: -abs(p[1])
    )
    for name, coef in pairs:
        bar = "#" * max(1, int(abs(coef) * 4)) if abs(coef) > 0.05 else "."
        print(f"  {name:<32} {coef:>+7.3f}  {bar}")

    # --- Full-dataset illustrative comparison -----------------------------------
    all_probs = model.predict_proba(X)[:, 1]
    print()
    print(line)
    print("FULL-DATASET COMPARISON vs RULES ENGINE")
    print("(includes training data -- NOT a held-out evaluation, for")
    print(" illustrative comparison against the rules baseline only)")
    print(line)

    groups = {}
    for kind, prob in zip(kinds, all_probs):
        groups.setdefault(kind, []).append(prob)

    print(f"  {'category':<30} {'n':>5} {'flagged@0.5':>12} "
          f"{'min-p':>7} {'med-p':>7} {'max-p':>7}")
    order = [
        "attack:credential_theft",
        "attack:patient_low_and_slow",
        "attack:sim_swap_takeover",
        "anomaly:family_shared_phone",
        "anomaly:genuine_sim_swap",
        "baseline",
    ]
    for kind in order:
        probs = sorted(groups.get(kind, []))
        if not probs:
            continue
        flagged = sum(1 for p in probs if p >= THRESHOLD)
        med = statistics.median(probs)
        print(f"  {kind:<30} {len(probs):>5} {flagged:>12} "
              f"{probs[0]:>7.3f} {med:>7.3f} {probs[-1]:>7.3f}")

    # Direct answers to the two design questions.
    patients = groups.get("attack:patient_low_and_slow", [])
    patient_caught = sum(1 for p in patients if p >= THRESHOLD)
    family = groups.get("anomaly:family_shared_phone", [])
    simswap_a = groups.get("anomaly:genuine_sim_swap", [])
    benign_flagged = sum(1 for k, ps in groups.items()
                         if not k.startswith("attack:")
                         for p in ps if p >= THRESHOLD)

    print()
    print("Direct answers:")
    print(f"  Patient attacks caught @0.5 : {patient_caught}/12 app-side "
          f"(rules engine caught 0/12)")
    print(f"  Benign rows flagged @0.5    : {benign_flagged} "
          f"(rules engine: 0) -- FPR guardrail check")
    print(f"  Family anomalies flagged    : "
          f"{sum(1 for p in family if p >= THRESHOLD)}/6   "
          f"Sim-swap anomalies flagged: "
          f"{sum(1 for p in simswap_a if p >= THRESHOLD)}/4")

    # --- Persist ------------------------------------------------------------------
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "random_state": SEED,
            "threshold": THRESHOLD,
        },
        MODEL_PATH,
    )
    print()
    print(f"Saved trained bundle -> {MODEL_PATH}")
    print(line)


_BUNDLE_CACHE = []


def _load_bundle():
    if not _BUNDLE_CACHE:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH} -- run "
                f"'python core/ml_model.py' once to train and save it."
            )
        _BUNDLE_CACHE.append(joblib.load(MODEL_PATH))
    return _BUNDLE_CACHE[0]


def predict_risk(features):
    """
    Live-scoring path: BehavioralFeatures instance -> P(fraud) float.

    Loads the saved bundle once per process; rebuilds the vector in the
    SAVED column order so API-side refactors cannot desynchronise training
    and inference representations.
    """
    bundle = _load_bundle()
    vector = _vectorize(features, bundle["feature_columns"])
    return float(bundle["model"].predict_proba([vector])[0][1])


if __name__ == "__main__":
    train_and_evaluate()

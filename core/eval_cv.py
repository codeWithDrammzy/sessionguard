"""
SessionGuard honest cross-validation evaluation harness
=======================================================

Replaces the older in-sample / single-80-20 evaluation methodology with a
proper **repeated stratified k-fold cross-validation** guarded against the
biggest threat to trust: *leakage of held-out information into the features
of a test example*.

WHY THIS EXISTS (methodology fix, not a product change)
-------------------------------------------------------
The earlier reported numbers (100% precision across rules/ML/hybrid) were
produced by:
  * the rules & hybrid scorers running over the FULL labelled dataset
    (in-sample, no held-out split at all), and
  * the ML using ONE stratified 80/20 split with only ~8 test positives.

Both were honest-as-labelled but are NOT a defensible answer to "how did you
evaluate generalisation with only 42 positive examples?". This module runs
the de-leaked variant.

THE LEAK IT CLOSES
------------------
With exactly 42 positives, a plain random session-level fold split lets a
"test" session's feature vector quietly see OTHER test-fold sessions:
  * history features (device/sim/location/velocity/keystroke/menu timing/
    impossible-travel) are causal in TIME, not in FOLD -- the stored
    BehavioralFeatures row for a test session was computed with ALL of that
    user's earlier sessions in history, including earlier ones that also
    landed in the test fold (`feature_engine.load_user_history` uses
    `timestamp__lt=` with no fold awareness).

The fix is methodology-only: for every fold each test session's features are
RECOMPUTED using a `UserHistory` built from the **training-fold sessions
only** (`compute_features(session, history=...)` supports injecting a
caller-owned history). Training features are likewise recomputed causally
within the train fold. Nothing about the rules weights, ML architecture or
hybrid logic changes.

RESIDUAL, DOCUMENTED LIMITATION (kept intentionally)
----------------------------------------------------
The `BankUser.typical_*` amount/hour baselines and per-transaction
`is_new_recipient` are attributes of the user profile set by the *dataset
generator* and reused to generate BOTH train and test data; they are not
session-"computed" so they are not fold-recomputed here. They are reported
as a known property of the synthetic-generation design (identical to the
live scoring path), not silently altered.

Metrics are mean +/- std across all folds, plus a confusion matrix summed
across folds:
  * precision, recall, F1,
  * Precision-Recall AUC (average_precision_score) -- the PRIMARY metric
    (accuracy/precision alone mislead at a ~2% positive rate),
  * summed confusion matrix,
  * the two safety checks: attacks caught (blocked-or-challenged) and
    genuine SIM-swap / family-sharing false-block rate.

No thresholds or features are tuned to push numbers up. The ML operating
point stays at p >= 0.5; the rules/hybrid band boundaries stay at 29 / 59.

Usage:
    python core/eval_cv.py [--splits 5] [--repeats 10] [--seed 46]
"""
import argparse
import os
import sys

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sessionguard_project.settings")
    import django

    django.setup()

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core.feature_engine import UserHistory, compute_features  # noqa: E402
from core.models import FraudLabel, Session  # noqa: E402
from core.ml_model import FEATURE_COLUMNS, _vectorize  # noqa: E402
from core.rules_engine import APPROVE_MAX, CHALLENGE_MAX, score_session  # noqa: E402
from core.hybrid_scorer import is_context_normal  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------


def _label_of(session, labels):
    return labels.get(session.session_id)


def _is_attack(session, labels):
    lab = labels.get(session.session_id)
    return bool(lab and lab["is_attack"])


def load_raw_sessions():
    labels = {
        row["session_id"]: row
        for row in FraudLabel.objects.values(
            "session_id", "is_attack", "attack_type", "is_legitimate_anomaly"
        )
    }
    sessions = list(
        Session.objects.select_related("user")
        .prefetch_related("transactions", "keystroke_dynamics")
        .order_by("user_id", "timestamp")
    )
    y = np.array([1 if _is_attack(s, labels) else 0 for s in sessions])
    kinds = {}
    for s in sessions:
        lab = labels.get(s.session_id)
        if lab is None:
            kinds[s.session_id] = "baseline"
        elif lab["is_attack"]:
            kinds[s.session_id] = f"attack:{lab['attack_type']}"
        elif lab["is_legitimate_anomaly"]:
            kinds[s.session_id] = (
                "anomaly:family_shared_phone"
                if not s.is_new_sim
                else "anomaly:genuine_sim_swap"
            )
        else:
            kinds[s.session_id] = "other"
    return sessions, labels, kinds, y


def _band(score):
    if score > CHALLENGE_MAX:
        return "block"
    if score > APPROVE_MAX:
        return "challenge"
    return "approve"


def _hybrid_verdict(features, ml_proba):
    """Reproduce hybrid_scorer logic with a fold-trained model's probability
    (the only intentional difference from production: the ML source)."""
    rules_decision = score_session(features)
    combined = max(rules_decision.score, round(float(ml_proba) * 100))
    verdict = _band(combined)
    if (
        verdict == "block"
        and (features.device_change_flag or features.sim_change_flag)
        and is_context_normal(features)
    ):
        verdict = "challenge"
    return verdict, combined


# ---------------------------------------------------------------------------
# One fold.
# ---------------------------------------------------------------------------


def evaluate_fold(sessions, labels, kinds, y, train_idx, test_idx):
    train_sessions = [sessions[i] for i in train_idx]
    test_sessions = [sessions[i] for i in test_idx]

    # --- Train features (causal within the train fold only) ------------------
    hist = {}
    X_train, y_train = [], []
    for s in train_sessions:
        h = hist.setdefault(s.user_id, UserHistory())
        ft = compute_features(s, history=h)
        X_train.append(_vectorize(ft, FEATURE_COLUMNS))
        y_train.append(1 if _is_attack(s, labels) else 0)
        h.observe(s)  # after scoring -> causality
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
    model.fit(X_train, y_train)

    # --- Test features: de-leaked (train-only history, never observed) ------
    # Group this user's train sessions by user, kept in timestamp order (the
    # global (user, timestamp) sort preserves it).
    per_user_train = {}
    for s in train_sessions:
        per_user_train.setdefault(s.user_id, []).append(s)

    def train_history_for(user_id, ts):
        """Strictly-causal train-only history for one test session: replay
        ONLY that user's train-fold sessions with timestamp < ts. This is
        the fold-aware equivalent of load_user_history()."""
        h = UserHistory()
        for s in per_user_train.get(user_id, ()):  # already ts-ascending
            if s.timestamp < ts:
                h.observe(s)
            else:
                break
        return h

    test_rows = []
    for s in test_sessions:
        ft = compute_features(s, history=train_history_for(s.user_id, s.timestamp))
        test_rows.append((ft, s))
    X_test = np.array([_vectorize(ft, FEATURE_COLUMNS) for ft, _ in test_rows])
    ml_proba = model.predict_proba(X_test)[:, 1]

    y_true = np.array([1 if _is_attack(s, labels) else 0 for _, s in test_rows])
    rules_v = [_band(score_session(ft).score) for ft, _ in test_rows]
    ml_v = ["block" if p >= 0.5 else "approve" for p in ml_proba]
    hybrid_pairs = [_hybrid_verdict(ft, p)
                    for ft, p in zip((r[0] for r in test_rows), ml_proba)]
    hybrid_v = [h for h, _ in hybrid_pairs]

    res = {}
    for name, verdicts in (("rules", rules_v), ("ml", ml_v), ("hybrid", hybrid_v)):
        y_hard = np.array([v == "block" for v in verdicts]).astype(int)
        cm = confusion_matrix(y_true, y_hard, labels=[0, 1])
        res[f"{name}_precision"] = precision_score(y_true, y_hard, zero_division=0)
        res[f"{name}_recall"] = recall_score(y_true, y_hard, zero_division=0)
        res[f"{name}_f1"] = f1_score(y_true, y_hard, zero_division=0)
        res[f"{name}_cm"] = cm
        res[f"{name}_caught"] = sum(
            1 for v, s in zip(verdicts, (s for _, s in test_rows))
            if v in ("block", "challenge") and _is_attack(s, labels)
        )
        res[f"{name}_npos"] = int(y_true.sum())
        res[f"{name}_simswap_fp"] = sum(
            1 for v, s in zip(verdicts, (s for _, s in test_rows))
            if kinds[s.session_id] == "anomaly:genuine_sim_swap" and v == "block"
        )
        res[f"{name}_nsimswap"] = sum(
            1 for _, s in test_rows if kinds[s.session_id] == "anomaly:genuine_sim_swap"
        )
        res[f"{name}_family_fp"] = sum(
            1 for v, s in zip(verdicts, (s for _, s in test_rows))
            if kinds[s.session_id] == "anomaly:family_shared_phone" and v == "block"
        )
        res[f"{name}_nfamily"] = sum(
            1 for _, s in test_rows if kinds[s.session_id] == "anomaly:family_shared_phone"
        )
        # PR-AUC: ML uses probabilities; rules/hybrid use the punitive score
        # as a continuous ranking proxy (higher = more confident fraud).
        if name == "ml":
            scores = ml_proba
        elif name == "rules":
            scores = np.array([float(score_session(ft).score) for ft, _ in test_rows])
        else:
            scores = np.array([s for _, s in hybrid_pairs])
        res[f"{name}_pr_auc"] = average_precision_score(y_true, scores)
    res["npos_total"] = int(y_true.sum())
    return res


# ---------------------------------------------------------------------------
# Aggregation + printing.
# ---------------------------------------------------------------------------


def _mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std())


def _fm(vals):
    m, s = _mean_std(vals)
    return f"{m:.3f} +/- {s:.3f}"


def _ratio(num, den):
    return (float(num) / float(den)) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=46)
    args = ap.parse_args()

    line = "=" * 78
    print(line)
    print("SESSIONGUARD HONEST CV EVALUATION "
          f"(RepeatedStratifiedKFold k={args.splits} r={args.repeats})")
    print(line)

    sessions, labels, kinds, y = load_raw_sessions()
    n_pos = int(y.sum())
    print(f"Dataset      : {len(sessions)} sessions, {n_pos} positives "
          f"({100.0 * y.mean():.2f}% positive rate)")
    print(f"De-leak mode : test features recomputed from TRAIN-fold history "
          f"only (no test-to-test history leak)")
    print()

    rskf = RepeatedStratifiedKFold(
        n_splits=args.splits, n_repeats=args.repeats, random_state=args.seed
    )

    names = ("rules", "ml", "hybrid")
    agg = {n: {"precision": [], "recall": [], "f1": [], "pr_auc": [],
               "caught": [], "npos": [], "simswap_fp": [], "nsimswap": [],
               "family_fp": [], "nfamily": [], "cm": None}
           for n in names}

    fold_count = 0
    for train_idx, test_idx in rskf.split(np.zeros(len(sessions)), y):
        res = evaluate_fold(sessions, labels, kinds, y, train_idx, test_idx)
        fold_count += 1
        for n in names:
            a = agg[n]
            a["precision"].append(res[f"{n}_precision"])
            a["recall"].append(res[f"{n}_recall"])
            a["f1"].append(res[f"{n}_f1"])
            a["pr_auc"].append(res[f"{n}_pr_auc"])
            a["caught"].append(res[f"{n}_caught"])
            a["npos"].append(res[f"{n}_npos"])
            a["simswap_fp"].append(res[f"{n}_simswap_fp"])
            a["nsimswap"].append(res[f"{n}_nsimswap"])
            a["family_fp"].append(res[f"{n}_family_fp"])
            a["nfamily"].append(res[f"{n}_nfamily"])
            if a["cm"] is None:
                a["cm"] = np.zeros((2, 2), dtype=int)
            a["cm"] += res[f"{n}_cm"]

    n_folds = fold_count
    print()
    print("PRIMARY / CORE METRICS  (mean +/- std over "
          f"{n_folds} folds = {args.splits} x {args.repeats})")
    hdr = (f"{'scorer':<8} {'precision':>15} {'recall':>15} {'F1':>15} "
           f"{'PR-AUC':>15}")
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        a = agg[n]
        print(f"{n:<8} {_fm(a['precision']):>15} {_fm(a['recall']):>15} "
              f"{_fm(a['f1']):>15} {_fm(a['pr_auc']):>15}")

    print()
    print("SUMMED CONFUSION MATRIX across all folds "
          "(hard-block = positive prediction; rows=actual [0,1], "
          "cols=pred [0,1]):")
    for n in names:
        cm = agg[n]["cm"]
        tp, fp = int(cm[1, 1]), int(cm[0, 1])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + int(cm[1, 0])) if (tp + int(cm[1, 0])) else 0.0
        print(f"  {n:<8} TN={int(cm[0,0]):<6} FP={fp:<6} FN={int(cm[1,0]):<6} "
              f"TP={tp:<6}  precision={prec:.4f}  recall={rec:.4f}")

    print()
    print("POOLED SAFETY CHECKS (summed across folds):")
    for n in names:
        a = agg[n]
        caught = int(sum(a["caught"]))
        npos = int(sum(a["npos"]))
        sswap = int(sum(a["simswap_fp"]))
        nsswap = int(sum(a["nsimswap"]))
        fam = int(sum(a["family_fp"]))
        nfam = int(sum(a["nfamily"]))
        print(f"  {n:<8} attacks-caught={caught}/{npos} "
              f"({_ratio(caught, npos)*100:.1f}%)   "
              f"sim-swap false-block={sswap}/{nsswap}   "
              f"family false-block={fam}/{nfam}")

    print()
    print("NOTES")
    print("- PR-AUC (average precision) is the PRIMARY metric; at a ~2%")
    print("  positive rate, precision/accuracy alone are misleading.")
    print("- 'caught' = block OR challenge (both stop the transfer until")
    print("  step-up verification passes); confusion matrix counts only hard")
    print("  'block' as positive.")
    print("- Each test session's features were recomputed with ONLY training")
    print("  history -- the de-leak guarantee.")
    print("- Residual: BankUser.typical_* amount/hour baselines remain the")
    print("  generator-set profile (shared by train and test by design).")
    print(line)


if __name__ == "__main__":
    main()

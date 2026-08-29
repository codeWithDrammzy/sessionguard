"""
SessionGuard hybrid scorer -- rules engine + ML model + context override
========================================================================

The single decision function the future API endpoint calls. Combines:

  * ``rules_engine.score_session``   -- transparent weighted signals
  * ``ml_model.predict_risk``        -- learned P(fraud) over the same features

...then applies ONE principled correction for the failure mode observed
when the two were evaluated separately:

WHY THE OVERRIDE EXISTS
-----------------------
Trained on this dataset, the ML model learned "device/SIM changed => fraud"
so aggressively (coefficients +6.6/+5.9) that genuine_sim_swap anomalies --
real customers recovering from a lost phone -- scored p>=0.94 and would be
blocked 4/4. The rules engine approved them but was blind to patient
attacks. The insight: HARDWARE CHANGE IS AMBIGUOUS ON ITS OWN. Its meaning
depends on whether everything ELSE looks ordinary:

  * patient attacker  -> changes hardware, careful elsewhere (that is WHY
                         they are patient) -> deserves interception;
  * genuine recovery  -> changes hardware, everything else IS their normal
                         life -> must never be hard-blocked.

We cannot separate these perfectly from behaviour, so we encode a
deliberate, documented compromise: when hardware changed but amount and
hour are unremarkable, cap severity at CHALLENGE (step-up verification).
Patient attacks still get caught -- with friction instead of a hard block;
a wrongly-suspected genuine customer faces an OTP prompt, not a frozen
account. Only "block" is softened, only by one level, never the reverse.

WHY MAX() COMBINES THE TWO SIGNALS
----------------------------------
The combined score takes MAX(rules_score, ml_prob * 100), not an average.
Averaging dilutes a strong true-positive signal from either side: rules=15
(quiet patient attack) + ML=90 would average to ~52 (challenge) when the
model is effectively certain; max keeps it at 90. Each engine's certainty
should be able to carry the decision alone.

EVALUATION CAVEAT (same as ml_model.py): the batch report runs over the
FULL dataset, and the ML component was trained on 80% of it -- illustrative
comparison against the rules baseline, not a clean generalisation metric.

Usage:
    python core/hybrid_scorer.py                 # full evaluation report
    from core.hybrid_scorer import score_session_hybrid   # live API path
"""

import os
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

from dataclasses import dataclass  # noqa: E402

from core.ml_model import predict_risk  # noqa: E402
from core.models import BehavioralFeatures, FraudLabel  # noqa: E402
from core.rules_engine import (  # noqa: E402
    APPROVE_MAX,
    CHALLENGE_MAX,
    RiskDecision,
    score_session,
)

# ---------------------------------------------------------------------------
# Context-normalcy configuration -- named, documented, calibratable.
# ---------------------------------------------------------------------------
CONTEXT_NORMAL_THRESHOLDS = {
    # amount_deviation_score <= 0.15 means "within ~15% overshoot of the
    # user's typical ceiling" -- i.e. an ordinary-sized transfer for them.
    "amount_deviation_max": 0.15,
    # hour_deviation_score <= 0.15 means "at most ~54 minutes outside the
    # user's declared windows" -- i.e. ordinary schedule jitter.
    "hour_deviation_max": 0.15,
    # NOTE: new_recipient_flag is deliberately NOT part of normalcy scoring.
    # We established elsewhere (WEIGHTS["new_recipient_alone"]=3) that paying
    # someone new alone is routine life; counting it here would erode the
    # override for exactly the harmless cases it exists to protect.
    #
    # NOTE: impossible travel IS part of normalcy scoring -- a leap to a
    # city that could not be physically reached since the last session is
    # the opposite of an otherwise-ordinary session, so the block->challenge
    # hardware override must never soften it.
    "impossible_travel_blocks": True,
}


def is_context_normal(features):
    """
    True when NOTHING ABOUT THE SESSION'S BEHAVIOUR besides the hardware
    change looks unusual for this user.

    This function's entire purpose is separating "changed hardware in an
    otherwise-ordinary session" (family phone, genuine SIM-swap recovery)
    from "changed hardware as part of a broader attack pattern" (unusual
    amounts, unusual hours). A null amount_deviation_score counts as
    normal: a session with no transaction has nothing abnormal to measure.
    """
    amount = features.amount_deviation_score
    if (
        amount is not None
        and amount > CONTEXT_NORMAL_THRESHOLDS["amount_deviation_max"]
    ):
        return False
    if features.hour_deviation_score > CONTEXT_NORMAL_THRESHOLDS[
        "hour_deviation_max"
    ]:
        return False
    if CONTEXT_NORMAL_THRESHOLDS["impossible_travel_blocks"] and (
        features.impossible_travel_flag
    ):
        return False
    return True


@dataclass
class HybridDecision(RiskDecision):
    """RiskDecision plus the two fields needed for audit/explanation."""

    ml_probability: float = 0.0
    context_override_applied: bool = False


def _band(score):
    """SAME bands as rules_engine (constants imported, not redefined)."""
    if score > CHALLENGE_MAX:
        return "block"
    if score > APPROVE_MAX:
        return "challenge"
    return "approve"


def score_session_hybrid(features):
    """
    Final production decision for one session.

    Steps: rules score -> ML probability -> combined = max(...) -> band ->
    context-normalcy override (block->challenge ONLY) -> merged reasons.
    """
    rules_decision = score_session(features)
    ml_prob = predict_risk(features)
    ml_points = round(ml_prob * 100)

    combined = max(rules_decision.score, ml_points)
    verdict = _band(combined)

    # THE OVERRIDE: hardware changed + otherwise-ordinary session =>
    # cap at challenge. Never applied upward, never to approve/challenge.
    context_override_applied = False
    if (
        verdict == "block"
        and (features.device_change_flag or features.sim_change_flag)
        and is_context_normal(features)
    ):
        verdict = "challenge"
        context_override_applied = True

    triggered_reasons = list(rules_decision.triggered_reasons)
    if ml_points >= rules_decision.score:
        # ML was the binding constraint (or tied): record its contribution.
        triggered_reasons.append(
            {"code": "ml_model_risk", "weight": ml_points}
        )
    if context_override_applied:
        triggered_reasons.append(
            {"code": "context_normal_override", "weight": 0}
        )

    return HybridDecision(
        session_id=str(features.session_id),
        score=combined,
        verdict=verdict,
        triggered_reasons=triggered_reasons,
        ml_probability=ml_prob,
        context_override_applied=context_override_applied,
    )


# ---------------------------------------------------------------------------
# Batch evaluation: hybrid report + three-way comparison
# ---------------------------------------------------------------------------

def _pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def _metrics_from(decisions, kinds):
    """TP/FP/FN/TN treating 'block' as the positive prediction.

    Accepts EITHER decision objects (.verdict) or plain verdict strings --
    the three-way comparison passes strings, and a non-empty string must
    never silently count as truthy 'blocked'.
    """
    tp = fp = fn = tn = 0
    for d, kind in zip(decisions, kinds):
        if isinstance(d, str):
            blocked = d == "block"
        else:
            blocked = d.verdict == "block"
        attacked = kind.startswith("attack:")
        if attacked and blocked:
            tp += 1
        elif attacked:
            fn += 1
        elif blocked:
            fp += 1
        else:
            tn += 1
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def score_all_sessions_hybrid():
    line = "=" * 78

    rows = []  # (kind, rules_dec, ml_flagged, hybrid_dec)
    labels = {
        r["session_id"]: r
        for r in FraudLabel.objects.values(
            "session_id", "is_attack", "attack_type", "is_legitimate_anomaly"
        )
    }
    for f in BehavioralFeatures.objects.select_related(
        "session"
    ).iterator(chunk_size=500):
        label = labels.get(f.session_id)
        if label is None:
            kind = "baseline"
        elif label["is_attack"]:
            kind = f"attack:{label['attack_type']}"
        elif label["is_legitimate_anomaly"]:
            kind = (
                "anomaly:family_shared_phone"
                if not f.session.is_new_sim
                else "anomaly:genuine_sim_swap"
            )
        else:
            kind = "other"

        rules_dec = score_session(f)
        ml_flagged = predict_risk(f) >= 0.5
        hybrid_dec = score_session_hybrid(f)
        rows.append((kind, rules_dec, ml_flagged, hybrid_dec))

    decisions_rules = [r[1] for r in rows]
    ml_flags = [r[2] for r in rows]
    decisions_hybrid = [r[3] for r in rows]
    kinds = [r[0] for r in rows]

    # --- Full hybrid report ---------------------------------------------------
    m = _metrics_from(decisions_hybrid, kinds)
    print(line)
    print("HYBRID SCORER EVALUATION (full dataset -- ML half-trained on it;")
    print(" illustrative comparison against the rules baseline, see docstring)")
    print(line)
    print(f"Sessions scored : {len(rows)}")
    print(f"Confusion ('block'=positive): TP={m['tp']}  FP={m['fp']}  "
          f"FN={m['fn']}  TN={m['tn']}")
    print(f"Precision {_pct(m['tp'], m['tp'] + m['fp'])} | "
          f"Recall {_pct(m['tp'], m['tp'] + m['fn'])} | "
          f"FPR {_pct(m['fp'], m['fp'] + m['tn'])}")
    print()

    print("Attack breakdown (hybrid):")
    for kind in sorted({k for k in kinds if k.startswith("attack:")}):
        ds = [d for k, d in zip(kinds, decisions_hybrid) if k == kind]
        n = len(ds)
        b = sum(1 for d in ds if d.verdict == "block")
        c = sum(1 for d in ds if d.verdict == "challenge")
        a = sum(1 for d in ds if d.verdict == "approve")
        ov = sum(1 for d in ds if d.context_override_applied)
        print(f"  {kind:<28} n={n:>2}  block={b:>2} ({_pct(b, n)})  "
              f"challenge={c:>2}  approve={a:>2}  overrides={ov}")

    print()
    print("Legitimate anomalies (hybrid):")
    for name in ["anomaly:family_shared_phone", "anomaly:genuine_sim_swap"]:
        ds = [d for k, d in zip(kinds, decisions_hybrid) if k == name]
        n = len(ds)
        b = sum(1 for d in ds if d.verdict == "block")
        c = sum(1 for d in ds if d.verdict == "challenge")
        a = sum(1 for d in ds if d.verdict == "approve")
        print(f"  {name:<28} n={n:>2}  approve={a:>2}  "
              f"challenge={c:>2}  block={b:>2}")

    # --- THREE-WAY comparison ---------------------------------------------------
    def summarise(decision_or_flag_seq):
        decs = []
        for item in decision_or_flag_seq:
            if isinstance(item, bool):
                decs.append("block" if item else "approve")  # ML-only view
            else:
                decs.append(item.verdict)
        return decs

    views = {
        "rules-only": summarise(decisions_rules),
        "ML-only @0.5": summarise(ml_flags),
        "hybrid": summarise(decisions_hybrid),
    }

    print()
    print(line)
    print("THREE-WAY COMPARISON (the decisive table)")
    print(line)
    hdr = (f"{'scorer':<14} {'precision':>9} {'recall':>8} {'FPR':>7} | "
           f"{'patient strict':>14} {'patient op.':>11} | "
           f"{'simswap FP':>10}")
    print(hdr)

    patient_kinds_idx = [
        i for i, k in enumerate(kinds)
        if k == "attack:patient_low_and_slow"
    ]
    simswap_idx = [
        i for i, k in enumerate(kinds)
        if k == "anomaly:genuine_sim_swap"
    ]

    for name, decs in views.items():
        met = _metrics_from(decs, kinds)
        p_strict = sum(1 for i in patient_kinds_idx if decs[i] == "block")
        p_op = sum(
            1 for i in patient_kinds_idx if decs[i] in ("block", "challenge")
        )
        s_fp = sum(1 for i in simswap_idx if decs[i] == "block")
        print(f"{name:<14} {_pct(met['tp'], met['tp'] + met['fp']):>9} "
              f"{_pct(met['tp'], met['tp'] + met['fn']):>8} "
              f"{_pct(met['fp'], met['fp'] + met['tn']):>7} | "
              f"{f'{p_strict}/12':>14} {f'{p_op}/12':>11} | "
              f"{f'{s_fp}/4':>10}")
    print("-" * 78)
    print("'patient strict' = hard-blocked; 'patient op.' = blocked OR")
    print("challenged (friction still stops the transfer until step-up")
    print("verification passes). 'simswap FP' = genuine post-loss SIM-swap")
    print("recoveries HARD-BLOCKED -- must be 0.")
    print(line)


if __name__ == "__main__":
    score_all_sessions_hybrid()

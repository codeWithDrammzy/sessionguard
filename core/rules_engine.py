"""
SessionGuard weighted-scoring rules engine
==========================================

Turns a session's BehavioralFeatures into a verdict -- approve /
challenge / block -- plus plain-English trigger reasons. Pattern adapted
from the Riskora fraud-scoring approach: weighted signals summed into a
score, banded into a verdict.

THIS IS THE BASELINE, NOT THE FINISHED PRODUCT
----------------------------------------------
Every weight below is HAND-TUNED from first principles (how damning each
signal is on its own), deliberately BEFORE looking at evaluation results,
so the numbers in score_all_sessions()'s report are an honest baseline for
comparison. They are calibratable against evaluation data later, and the
gap between this rules engine and a future ML model is exactly what our
write-up should quantify. No peeking, no tweaking until the report exists.

DESIGN NOTES
------------
* ``combined_device_location`` subsumes its components: when it fires, the
  device/SIM/location "alone" rules do NOT also score -- no double
  counting of one underlying event.
* ``new_recipient_alone`` is worth a deliberately tiny +3: the brief warns
  that paying someone new is normal life; alone it must never move a
  verdict, it only adds weight in combination.
* Velocity scores sessions BEYOND the first in the 5-minute window
  (max(0, count - 1) * 8): 0 or 1 transactions is normal, so the literal
  ``count * 8`` reading would punish ordinary single transfers.
* Scaled rules (hour/amount/menu-timing) contribute round(score * max).

Usage:
    python core/rules_engine.py              # batch evaluation report
    from core.rules_engine import score_session        # live scoring path
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

from dataclasses import dataclass, field  # noqa: E402

from core.models import BehavioralFeatures, FraudLabel  # noqa: E402


# ---------------------------------------------------------------------------
# Scoring configuration -- the whole logic, readable in one place.
# Hand-tuned starting weights (see module docstring), NOT empirically optimal.
# ---------------------------------------------------------------------------
WEIGHTS = {
    # Strongest single signal: credential context AND location moved together.
    "combined_device_location": 45,
    # Weak evidence alone: new phone, shared tablet, routine SIM upgrade...
    "device_change_alone": 10,
    "sim_change_alone": 10,
    "location_change_alone": 5,
    # Scaled rules: contribution = feature_score * max_points.
    "hour_deviation_max": 15,
    "amount_deviation_max": 20,
    "menu_timing_deviation_max": 15,
    # Per EXTRA transaction beyond the first inside the 5-minute window.
    "velocity_per_extra_session": 8,
    # Deliberately tiny: new recipient alone is normal life (brief warning).
    "new_recipient_alone": 3,
}

APPROVE_MAX = 29  # 0-29   -> approve
CHALLENGE_MAX = 59  # 30-59 -> challenge (step-up verification: OTP etc.)
# >= 60                     -> block


@dataclass
class RiskDecision:
    """Outcome of scoring one session."""

    session_id: str
    score: int  # 0-100+ (uncapped: extreme sessions can exceed 100)
    verdict: str  # "approve" | "challenge" | "block"
    triggered_reasons: list = field(default_factory=list)  # [{"code","weight"}]


def _band(score):
    if score >= 60:
        return "block"
    if score >= 30:
        return "challenge"
    return "approve"


def score_session(features):
    """
    Score ONE BehavioralFeatures row into a RiskDecision.

    Pure function of the feature vector -- no DB access -- so the live API
    can call it immediately after compute_features() with zero extra I/O.
    """
    reasons = []
    total = 0

    def add(code, points):
        nonlocal total
        if points > 0:
            reasons.append({"code": code, "weight": points})
            total += points

    # Context-change cluster: combined flag subsumes its parts.
    if features.combined_device_location_flag:
        add("combined_device_location", WEIGHTS["combined_device_location"])
    else:
        if features.device_change_flag:
            add("device_change_alone", WEIGHTS["device_change_alone"])
        if features.sim_change_flag:
            add("sim_change_alone", WEIGHTS["sim_change_alone"])
        if features.location_change_flag:
            add("location_change_alone", WEIGHTS["location_change_alone"])

    # Scaled behavioural signals.
    if features.hour_deviation_score:
        add(
            "hour_deviation",
            round(features.hour_deviation_score * WEIGHTS["hour_deviation_max"]),
        )
    if features.amount_deviation_score is not None:
        add(
            "amount_deviation",
            round(
                features.amount_deviation_score
                * WEIGHTS["amount_deviation_max"]
            ),
        )
    if features.menu_timing_deviation_score is not None:
        add(
            "menu_timing_deviation",
            round(
                features.menu_timing_deviation_score
                * WEIGHTS["menu_timing_deviation_max"]
            ),
        )

    # Velocity: only sessions BEYOND the first count (see module docstring).
    extras = max(0, features.velocity_count_5min - 1)
    if extras:
        add("velocity_burst", extras * WEIGHTS["velocity_per_extra_session"])

    if features.new_recipient_flag:
        add("new_recipient", WEIGHTS["new_recipient_alone"])

    return RiskDecision(
        session_id=str(features.session_id),
        score=total,
        verdict=_band(total),
        triggered_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Batch evaluation over the labelled dataset
# ---------------------------------------------------------------------------

def _verdict_counts(decisions):
    c = {"approve": 0, "challenge": 0, "block": 0}
    for d in decisions:
        c[d.verdict] += 1
    return c


def _pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def score_all_sessions():
    """
    Score every session's stored features and print the full evaluation
    report. Nothing is persisted -- decisions live in memory only.

    Label semantics: FraudLabel absent -> baseline (negative); is_attack ->
    positive; is_legitimate_anomaly -> hard negative (must not be blocked).
    "challenge" is reported separately everywhere as an ambiguous middle.
    """
    attacks, baselines, family_anoms, simswap_anoms = [], [], [], []

    for f in BehavioralFeatures.objects.select_related(
        "session__fraud_label"
    ):
        decision = score_session(f)
        try:
            label = f.session.fraud_label
        except FraudLabel.DoesNotExist:
            label = None

        if label is None:
            baselines.append(decision)
        elif label.is_attack:
            attacks.append((decision, label.attack_type))
        elif label.is_legitimate_anomaly:
            if f.session.is_new_sim:
                simswap_anoms.append(decision)
            else:
                family_anoms.append(decision)

    _print_report(attacks, baselines, family_anoms, simswap_anoms)


def _print_report(attacks, baselines, family_anoms, simswap_anoms):
    line = "=" * 70

    # --- Confusion matrix (block = positive prediction) ---------------------
    tp_pairs = [d for d, _ in attacks if d.verdict == "block"]
    fn_attacks = [(d, t) for d, t in attacks if d.verdict != "block"]
    fp_negatives = [
        d for d in baselines + family_anoms + simswap_anoms
        if d.verdict == "block"
    ]
    tn_negatives = len(baselines) + len(family_anoms) + len(simswap_anoms) - len(
        fp_negatives
    )
    challenge_attacks = sum(1 for d, _ in fn_attacks if d.verdict == "challenge")

    precision = (
        len(tp_pairs) / (len(tp_pairs) + len(fp_negatives))
        if (tp_pairs or fp_negatives)
        else 0.0
    )
    recall = len(tp_pairs) / len(attacks) if attacks else 0.0
    negatives = len(fp_negatives) + tn_negatives
    fpr = len(fp_negatives) / negatives if negatives else 0.0

    print()
    print(line)
    print("RULES ENGINE EVALUATION REPORT (hand-tuned weights, seed-fixed data)")
    print(line)
    print(f"Sessions scored               : "
          f"{len(attacks) + negatives} "
          f"(attacks={len(attacks)}, baseline={len(baselines)}, "
          f"anomalies={len(family_anoms) + len(simswap_anoms)})")
    print()
    print("Confusion matrix ('block' = positive prediction):")
    print(f"  True positives  (attack -> block)       : {len(tp_pairs)}"
          f" / {len(attacks)}")
    print(f"  False negatives (attack missed)         : {len(fn_attacks)}"
          f"  [of which 'challenge' (soft catch): {challenge_attacks}]")
    print(f"  False positives (benign -> block)       : {len(fp_negatives)}")
    print(f"  True negatives (benign not blocked)     : {tn_negatives}")
    print()
    print("Metrics:")
    print(f"  Precision : {_pct(len(tp_pairs), len(tp_pairs) + len(fp_negatives))}"
          f"   Recall: {_pct(len(tp_pairs), len(attacks))}   "
          f"FPR: {_pct(len(fp_negatives), negatives)}")
    # Step-up verification (challenge) still stops an attack operationally;
    # this metric counts those soft catches alongside hard blocks.
    operational_catch = len(tp_pairs) + challenge_attacks
    print(f"  Operational catch rate (block+challenge): "
          f"{_pct(operational_catch, len(attacks))}")
    print()

    # --- Per attack type ----------------------------------------------------
    print("Breakdown by attack_type (the honest difficulty story):")
    types = sorted({t for _, t in attacks})
    for attack_type in types:
        ds = [d for d, t in attacks if t == attack_type]
        c = _verdict_counts(ds)
        caught = c["block"] + c["challenge"]
        print(f"  {attack_type:<22} n={len(ds):>2}  "
              f"block={c['block']:>2} ({_pct(c['block'], len(ds))})  "
              f"block+challenge={_pct(caught, len(ds))}  "
              f"approve={c['approve']:>2}")

    # --- Legitimate anomalies -------------------------------------------------
    print()
    print("Legitimate anomalies (MUST skew approve/challenge):")
    for name, group in [("family_shared_phone", family_anoms),
                        ("genuine_sim_swap", simswap_anoms)]:
        c = _verdict_counts(group)
        print(f"  {name:<22} n={len(group):>2}  "
              f"approve={c['approve']:>2}  challenge={c['challenge']:>2}  "
              f"block={c['block']:>2}")
    blocked_family = [d for d in family_anoms if d.verdict == "block"]
    if blocked_family:
        print(f"  !!! FAILURE MODE: {len(blocked_family)} family_shared_phone "
              f"session(s) BLOCKED -- exactly the false-positive the brief "
              f"warns against. Reasons: "
              f"{[d.triggered_reasons for d in blocked_family]}")
    else:
        print("  OK: zero family_shared_phone blocks.")

    print()
    print("(Weights are documented starting points -- see WEIGHTS and module")
    print(" docstring. This report is the ML-comparison baseline.)")
    print(line)


if __name__ == "__main__":
    score_all_sessions()

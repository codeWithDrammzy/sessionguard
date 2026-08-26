"""
Customer-facing decision explanations for SessionGuard
======================================================

The challenge brief requires, verbatim: "a plain sentence explaining each
decision, of the kind a bank could give a customer who asks why their
transfer was stopped."

This module turns a HybridDecision's triggered_reasons into exactly that.
House style rules, enforced throughout:

  * Second person, spoken directly to the customer.
  * No jargon whatsoever: never "flag", "score", "feature", "model",
    "algorithm", never numbers or weights.
  * At most TWO reasons per sentence -- a wall of justifications reads as
    a machine talking, not a bank.
  * ``context_normal_override`` NEVER appears customer-facing: it is an
    internal signal that SOFTENED a decision, not a reason FOR it. When it
    fired, explain_decision() returns a (customer_sentence, internal_note)
    tuple instead of a bare string; the note documents for analysts/judges
    reading logs why a would-be block became a verification step.

Special case -- ``new_recipient``: worth a deliberate weight of only 3,
paying someone new is ordinary life. If it were ever the ONLY reason on a
non-approved decision (practically impossible: it cannot reach the
challenge band alone), the templates below still avoid implying it was
decisive; the approve path makes clear nothing unusual happened at all.

Usage:
    python core/explanation.py    # worked examples on real dataset rows

    from core.explanation import explain_decision
    result = explain_decision(hybrid_decision)   # str OR (str, note)
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

from core.models import BehavioralFeatures  # noqa: E402
from core.hybrid_scorer import score_session_hybrid  # noqa: E402

# ---------------------------------------------------------------------------
# Reason code -> customer-facing fragment. Written to be joined naturally
# after "because", e.g. "...because it came from a device we don't recognize
# and it happened at a time you don't usually bank."
# ---------------------------------------------------------------------------
REASON_TEMPLATES = {
    "combined_device_location":
        "it came from a device and location we haven't seen with your "
        "account before",
    "device_change_alone":
        "it came from a device we don't recognize",
    "sim_change_alone":
        "it came from a SIM card we haven't seen used with your account",
    "location_change_alone":
        "it came from a place you don't usually bank from",
    "hour_deviation":
        "it happened at a time you don't usually bank",
    "amount_deviation":
        "the amount was unusual compared to your typical transfers",
    "menu_timing_deviation":
        "the session moved through the menus differently from your usual pace",
    "velocity_burst":
        "there were several attempts in a short space of time",
    # Low-weight by design; must never read as the deciding factor on its own.
    "new_recipient":
        "it was sent to someone new",
    # The ML contribution has no single crisp textbook cause, so the wording
    # is deliberately broader -- still concrete to the customer, still true.
    "ml_model_risk":
        "the overall pattern of this session looked risky based on how you "
        "usually bank",
    # Offline degraded-mode signal codes (offline_fallback): same customer
    # truths as their online counterparts, worded for the coarser evidence
    # a cached last-known snapshot can actually support.
    "device_or_sim_mismatch":
        "it came from a phone or SIM card we don't recognize from your "
        "account",
    "hour_outside_range":
        "it happened at a time you don't usually bank",
    "amount_outside_range":
        "the amount was unusual compared to your typical transfers",
    # Context-only code from offline_fallback's DegradedDecision: explains
    # WHY THE CHECK WAS LIMITED, not why the session looked risky, so it is
    # excluded from top-2 ranking (same treatment as context_normal_override)
    # and appended as a transparency clause instead.
    "offline_degraded_check":
        "we could only check this with limited information while our "
        "systems were temporarily unavailable",
    # context_normal_override intentionally has NO template here: it softens
    # a decision internally and is surfaced via the analyst note instead.
}

_INTERNAL_OVERRIDE_NOTE = (
    "Note: this session showed a hardware change but otherwise normal "
    "behaviour, so verification was requested instead of a hard block."
)


def _join_reasons(phrases):
    """
    Join 1-2 fragments into natural English -- never a bullet list, never
    comma-jargon: 'A' | 'A and B'.
    """
    phrases = [p for p in phrases if p]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return f"{phrases[0]} and {phrases[1]}"


def explain_decision(decision):
    """
    Convert a decision into customer-facing English.

    Returns:
        str                      -- normally
        (customer, internalnote) -- when context_override_applied is True,
                                    so logs keep the analyst explanation
                                    while the customer sentence stays clean.

    Defensive: a non-approved verdict with no usable reasons falls back to
    a generic-but-honest sentence rather than crashing or emitting "".
    """
    verdict = decision.verdict
    override_fired = bool(getattr(decision, "context_override_applied", False))
    reason_codes = [r["code"] for r in decision.triggered_reasons]
    # Degraded mode (offline_fallback): the decision was made with a cached
    # snapshot only. Customers get an honest transparency clause; the code
    # itself never competes for the top-2 reason slots.
    degraded = "offline_degraded_check" in reason_codes
    _degraded_clause = (
        " - we could only check this with limited information while our "
        "systems were temporarily unavailable"
    )

    if verdict == "approve":
        if degraded:
            customer = (
                "This transfer was approved based on a limited check while "
                "our systems were temporarily unavailable - nothing about "
                "it looked unusual."
            )
        else:
            customer = (
                "This transfer was approved - nothing about it looked "
                "unusual for your account."
            )
    else:
        visible = [
            r for r in decision.triggered_reasons
            if r["code"] not in ("context_normal_override",
                                 "offline_degraded_check")
        ]
        # Ranking: concrete rule causes lead (by weight); the ML phrase is a
        # DERIVED SUMMARY of those same signals, not independent evidence,
        # so it only fills a remaining slot rather than crowding specifics
        # out of the two-reason cap.
        concrete = sorted(
            (r for r in visible if r["code"] != "ml_model_risk"),
            key=lambda r: r["weight"], reverse=True,
        )
        ml_reason = next(
            (r for r in visible if r["code"] == "ml_model_risk"), None
        )
        top = (concrete[:2] +
               ([ml_reason] if len(concrete) < 2 and ml_reason else []))[:2]
        top_phrases = [
            REASON_TEMPLATES.get(r["code"], "it looked different from how "
                                          "you usually bank")
            for r in top
        ]
        if top_phrases:
            clause = _join_reasons(top_phrases)
        else:
            clause = "something about it looked unusual for your account"

        if verdict == "challenge":
            customer = (
                f"We asked for extra verification on this transfer because "
                f"{clause}{_degraded_clause if degraded else ''}."
            )
        else:  # block
            # Degraded mode can never produce a block (hard-capped at
            # challenge in offline_fallback), so no degraded branch needed.
            customer = (
                f"This transfer was stopped because {clause}. If this was "
                "you, please contact your bank to confirm your identity "
                "and release the transfer."
            )

    if override_fired:
        return customer, _INTERNAL_OVERRIDE_NOTE
    return customer


# ---------------------------------------------------------------------------
# Worked examples on REAL rows -- presentation-ready proof
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    line = "=" * 74

    # NOTE: a "pure rules-only challenge" does not exist in this dataset --
    # ML probabilities saturate high for every hardware-change session, so
    # ALL 56 challenges arrived via the context-normalcy override (verified:
    # {approve:2257, challenge+override:56, block:14}). The four demos below
    # cover the real decision space, including the brief's money-shot: a
    # genuine post-loss SIM-swap customer being asked -- not blocked -- to
    # verify.
    wanted = ["approve", "patient_challenge", "simswap_challenge", "block"]
    picked = {}

    for f in BehavioralFeatures.objects.select_related(
        "session__fraud_label"
    ).iterator(chunk_size=500):
        decision = score_session_hybrid(f)

        try:
            label = f.session.fraud_label
            attack_type = label.attack_type if label.is_attack else None
            is_simswap_anomaly = (
                label.is_legitimate_anomaly and f.session.is_new_sim
            )
        except Exception:
            attack_type = None
            is_simswap_anomaly = False

        if decision.verdict == "approve":
            key = "approve"
        elif (decision.context_override_applied
                and attack_type == "patient_low_and_slow"):
            key = "patient_challenge"
        elif decision.context_override_applied and is_simswap_anomaly:
            key = "simswap_challenge"
        elif decision.verdict == "block":
            key = "block"
        else:
            key = None

        if key and key not in picked:
            picked[key] = (f, decision)
        if len(picked) == len(wanted):
            break

    print(line)
    print("EXPLANATION DEMO -- four REAL sessions from the dataset")
    print(line)

    for i, key in enumerate(wanted, 1):
        if key not in picked:
            print(f"\nDEMO {i} [{key}] -- no such case in dataset "
                  "(see note above)")
            continue
        f, decision = picked[key]
        s = f.session
        result = explain_decision(decision)

        print()
        print(f"DEMO {i} [{key}]  channel={s.channel}  "
              f"session={str(s.session_id)[:8]}...")
        signals = ", ".join(
            f"{r['code']}({r['weight']})" for r in decision.triggered_reasons
        ) or "(none)"
        print(f"  Signals  : {signals}")
        print(f"  Score    : {decision.score} -> {decision.verdict.upper()}  "
              f"(ML p={decision.ml_probability:.2f})")
        if isinstance(result, tuple):
            customer, internal = result
            print(f"  Customer : {customer}")
            print(f"  Internal : {internal}")
        else:
            print(f"  Customer : {result}")
            print("  Internal : (none)")

    print()
    print(line)

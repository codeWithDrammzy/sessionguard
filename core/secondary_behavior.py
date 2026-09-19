"""
Verified secondary behaviour pattern
====================================

A second, LOW-TRUST guard that sits AFTER the primary hybrid scorer and
consults only what the system itself has PROVEN -- not labels, not reviewer
judgements.

THE TRUST ANCHOR
----------------
`ConfirmedOutcome` is a human's word; `FraudLabel` is synthetic ground truth.
Neither is fit for a *live* secondary signal. The one fact the primary flow
establishes without anyone's opinion is step-up verification: a customer who
completes OTP for a challenged transfer was demonstrably reachable in control
of the account at that moment. Those verified moments are the ONLY seed for
this profile.

WHAT IT DOES
------------
For each successful verification we snapshot the session's behavioural
context (device / SIM / location / recipient / hour / amount) into a bounded
FIFO window (`VERIFIED_HISTORY_LIMIT`) on `SecondaryBehaviorProfile`. OTP-
verified APP sessions additionally contribute a derived typing-rhythm profile
(the same timings the primary engine learns, but learned ONLY from verified
evidence). When a later session is challenged, the secondary stage measures
how much of that challenged session's context matches what the customer has
genuinely been verified doing:

  * familiarity >= CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE -> the challenge is
    CONFIRMED: the customer's own verified behaviour is under suspicion, so
    keep the proportional step-up friction.
  * familiarity <  ESCALATE_IF_FAMILIARITY_BELOW         -> ESCALATE to a hard
    block: the challenged session matches NONE of the trusted verified
    pattern, so the "step-up the real customer" assumption fails.
  * between the thresholds -> leave the challenge alone (action "none").

KEYSTROKE CAP (typing can only REDUCE trust, never add it)
----------------------------------------------------------
Once the verified window carries enough typing evidence
(`MIN_KEYSTROKE_VERIFIED_SNAPSHOTS`), the challenged session's rhythm is
z-scored against the verified typist and inverted to a similarity. That
similarity CAPS the context familiarity: a foreign typist on an otherwise
fully verified device/SIM/location (a family member or an impostor on the
original account holder's phone) can never be CONFIRMED from context alone.
Sessions with no keystroke capture (USSD, login-only) are exempt -- for them
typing simply does not exist, and behaviour is byte-identical to before.

This profile learns keystroke behaviour INDEPENDENTLY of the primary engine's
5-to-20 session baseline: it knows only what OTP verification proved, and it
keeps its own floor (`MIN_KEYSTROKE_VERIFIED_SNAPSHOTS`), never borrowing the
primary profile's confidence.

TERMINOLOGY
-----------
"original account holder" = the customer whose verified behaviour filled this
profile; "alternate legitimate user" = a different person legitimately acting
on the same account (e.g. a family member on the same rooted device). The
design treats the two as distinct: context match alone does not prove the
original account holder is the one currently typing.

HARD GATE BOUNDS (never violated)
--------------------------------
The primary engines (`rules_engine`, `hybrid_scorer`) stay PURE and untouched.
This stage is an explicit wrapper attached to the online choke point and it
ONLY ever acts on a primary verdict of ``challenge``:

  * approve  -> NEVER consulted, verdict unchanged (no manufactured blocks).
  * block    -> NEVER softened (a block at the primary stage is final here).
  * challenge-> may be confirmed (stays), escalated one band to block, or
               left unchanged. Escalation is the ONLY mutation, and it is
               bounded: challenge -> block, nothing else.

NOT A WHITELIST
---------------
The verified window is small and sliding: a verified context ages out after
VERIFIED_HISTORY_LIMIT more recent verifications. Acquainting a NEW context
requires the customer to actually verify a session that presents it. There is
no permanent "trusted forever" device/SIM/location in this design.

DORMANT UNTIL CONFIGURED
------------------------
`profile.verification_count >= MIN_VERIFICATIONS_TO_CONFIGURE` (3). Below
that the gate is silent -- a system with thin proof does not act on it.
"""

from dataclasses import dataclass
from statistics import mean, pstdev

from django.db import transaction as db_transaction

from core.hybrid_scorer import HybridDecision

# ---------------------------------------------------------------------------
# Configuration -- named, documented, calibratable (same house style as the
# rules/hybrid constants).
# ---------------------------------------------------------------------------
# The profile must have recorded THIS many successful verifications before
# its signal may move a verdict. Thin proof stays silent.
MIN_VERIFICATIONS_TO_CONFIGURE = 3

# Bounded FIFO window (NOT a whitelist): the last N verified contexts count.
# Older verified behaviour ages out of the familiarity measurement.
VERIFIED_HISTORY_LIMIT = 12

# Familiarity thresholds on the 0..1 familiarity score.
ESCALATE_IF_FAMILIARITY_BELOW = 0.25
CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE = 0.50

# Minimum verified keystroke snapshots per metric before typing similarity is
# judged at all. A thinner baseline is too weak to differentiate anyone -- and
# the keystroke contribution stays silent (never punished) in that case.
MIN_KEYSTROKE_VERIFIED_SNAPSHOTS = 2

# Relative importance of each context dimension. Weights are normalised over
# the dimensions PRESENT in a session (missing dimensions are never punished:
# a USSD session simply has no device, a login-only session has no amount).
#
# NOTE: typing rhythm is intentionally NOT one of these weights. Keystroke
# similarity is applied as a separate CAP over the weighted score (see
# familiarity_score): a foreign rhythm can only pull familiarity DOWN, so a
# very different typist is never confirmed by an otherwise verified context.
DIMENSION_WEIGHTS = {
    "device": 0.25,
    "sim": 0.20,
    "location": 0.20,
    "recipient": 0.15,
    "hour": 0.10,
    "amount": 0.10,
}

# Hour similarity decays linearly to 0 this many hours of circular
# (midnight-aware) distance from a verified hour.
HOUR_SIMILARITY_SPAN_HOURS = 6.0

# Amount similarity: 1.0 at the verified amount, decays to 0 when the current
# amount is this many multiples of the verified amount away.
AMOUNT_SIMILARITY_MULTIPLIER = 2.0


@dataclass
class SecondaryDecision(HybridDecision):
    """A HybridDecision plus the secondary-stage audit fields."""

    secondary_profile_configured: bool = False
    secondary_verification_count: int = 0
    secondary_familiarity: float = None  # 0..1 once measured; None when dormant
    secondary_action: str = None  # None (dormant) | "none" | "escalate" | "confirm"


def _keystroke_features(session):
    """Derived typing-rhythm features of one session, or None when it has no
    keystroke capture (USSD / login-only / app rows without timing data)."""
    ks = session.keystroke_dynamics.first()
    if ks is None:
        return None
    return {
        "avg_hold_time_ms": ks.avg_hold_time_ms,
        "hold_time_std_ms": ks.hold_time_std_ms,
        "avg_interval_ms": ks.avg_interval_ms,
        "interval_std_ms": ks.interval_std_ms,
        "typing_speed_cpm": ks.typing_speed_cpm,
        "longest_pause_ms": ks.longest_pause_ms,
        "backspace_count": ks.backspace_count,
    }


def _keystroke_similarity(keystroke, contexts):
    """
    0..1 similarity of one session's typing rhythm against the OTP-verified
    window, or None while too thin to judge (never punished then).

    Per present metric, the session's value is z-scored against the verified
    snapshots' own stats for that metric (zero-variance verified rhythm: equal
    -> 0, any gap -> 3.0). The absolute z-scores combine as
    ``0.6 * max|z| + 0.4 * mean|z|``, that composite maps to a raw deviation
    which saturates at 3 SD, and the result is inverted -- an identical rhythm
    scores ~1.0 and a genuinely different typist (alternate legitimate user or
    impostor) scores ~0.
    """
    if not keystroke:
        return None
    verified = [c.get("keystroke") for c in contexts if c.get("keystroke")]
    z_scores = []
    for metric, current in keystroke.items():
        if current is None:
            continue
        vals = [v[metric] for v in verified if v.get(metric) is not None]
        if len(vals) < MIN_KEYSTROKE_VERIFIED_SNAPSHOTS:
            continue
        mu = mean(vals)
        sigma = pstdev(vals)
        if sigma > 0:
            z = (current - mu) / sigma
        elif current != mu:
            z = 3.0  # verified rhythm constant: any gap is maximally foreign
        else:
            z = 0.0
        z_scores.append(abs(z))
    if not z_scores:
        return None
    composite = 0.6 * max(z_scores) + 0.4 * (sum(z_scores) / len(z_scores))
    deviation = max(0.0, min(1.0, (composite - 1.0) / 2.0))
    return 1.0 - deviation


def _session_context(session):
    """
    Snapshot the behavioural context of one session, using dimension names
    that match DIMENSION_WEIGHTS. Values that are genuinely absent stay None
    (device on USSD, recipient/amount on a login-only session) so the
    familiarity measurement can exclude -- never punish -- them. The derived
    typing-rhythm features ride along under "keystroke".
    """
    txn = session.transactions.order_by("timestamp").first()
    return {
        "device": session.device_fingerprint,
        "sim": session.sim_id,
        "location": session.location_geohash,
        "recipient": txn.recipient_id if txn else None,
        "hour": session.timestamp.hour if session.timestamp else None,
        "amount": float(txn.amount) if txn else None,
        "keystroke": _keystroke_features(session),
    }


def _hour_similarity(timestamp, context_hour):
    """0..1 circular proximity of a timestamp's minute-of-day to a verified
    hour. 1.0 inside that hour, decaying to 0 over HOUR_SIMILARITY_SPAN_HOURS
    either way (midnight-aware, same spirit as feature_engine)."""
    if context_hour is None:
        return 0.0
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    c_minute = context_hour * 60 + 30  # middle of the verified hour
    d = abs(minute_of_day - c_minute) % 1440
    d = min(d, 1440 - d)
    span = HOUR_SIMILARITY_SPAN_HOURS * 60
    return max(0.0, 1.0 - d / span)


def _amount_similarity(amount, verified_amount):
    """0..1 similarity of one amount to a verified amount: 1.0 when equal,
    decaying to 0 at AMOUNT_SIMILARITY_MULTIPLIER * verified_amount away."""
    if not verified_amount:
        return 0.0
    distance = abs(amount - verified_amount)
    return max(0.0, 1.0 - distance / (verified_amount * AMOUNT_SIMILARITY_MULTIPLIER))


def familiarity_score(session, contexts):
    """
    How much of ``session``'s context matches the customer's verified
    behaviour. 0..1.

    For membership dimensions (device/SIM/location/recipient) the best match
    is simply "was this exact value ever verified". For the degree dimensions
    (hour, amount) it is the best continuous similarity across the verified
    window. Each dimension is weighted by DIMENSION_WEIGHTS and the sum is
    renormalised over the dimensions actually present in THIS session.
    """
    if not contexts:
        return 0.0
    ctx = _session_context(session)
    per_dim = {}
    for dim in DIMENSION_WEIGHTS:
        value = ctx.get(dim)
        if value is None:
            continue  # absent dimension: excluded, never punished
        if dim == "hour":
            per_dim[dim] = max(
                (_hour_similarity(session.timestamp, c.get("hour"))
                 for c in contexts if c.get("hour") is not None),
                default=0.0,
            )
        elif dim == "amount":
            per_dim[dim] = max(
                (_amount_similarity(value, c.get("amount"))
                 for c in contexts if c.get("amount") is not None),
                default=0.0,
            )
        else:
            per_dim[dim] = 1.0 if any(c.get(dim) == value for c in contexts) else 0.0
    total_weight = sum(DIMENSION_WEIGHTS[d] for d in per_dim)
    if total_weight <= 0:
        return 0.0
    weighted = (
        sum(DIMENSION_WEIGHTS[d] * s for d, s in per_dim.items())
        / total_weight
    )
    # KEYSTROKE CAP: when this session carried typing evidence AND the
    # verified window has enough typing history to judge it, the rhythm can
    # only reduce (never raise) familiarity. A very different typist on an
    # otherwise fully verified device/SIM/location must not be trusted as the
    # original account holder. Sessions without keystroke data are untouched.
    keystroke_sim = _keystroke_similarity(ctx.get("keystroke"), contexts)
    if keystroke_sim is not None:
        weighted = min(weighted, keystroke_sim)
    return weighted


def secondary_action_for(familiarity):
    """Map a familiarity score to the secondary action."""
    if familiarity < ESCALATE_IF_FAMILIARITY_BELOW:
        return "escalate"
    if familiarity >= CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE:
        return "confirm"
    return "none"


def record_verification(session):
    """
    Idempotently record a successful OTP verification and fold the session's
    context into the user's verified-behaviour profile.

    Returns ``(created, profile)``. ``created`` is False when the session was
    already verified (double-submitted release, replay) -- the count is never
    inflated and the context is never duplicated.
    """
    from core.models import SecondaryBehaviorProfile, SecondaryVerification

    with db_transaction.atomic():
        existing = SecondaryVerification.objects.filter(session=session).first()
        if existing is not None:
            profile = (
                SecondaryBehaviorProfile.objects.filter(user=session.user).first()
            )
            return False, profile
        SecondaryVerification.objects.create(session=session)
        profile, _ = SecondaryBehaviorProfile.objects.get_or_create(
            user=session.user
        )
        contexts = list(profile.verified_contexts or [])
        contexts.append(_session_context(session))
        profile.verified_contexts = contexts[-VERIFIED_HISTORY_LIMIT:]
        profile.verification_count += 1
        profile.save()
    return True, profile


def apply_secondary_behavior(session, decision):
    """
    The secondary stage attached to the online scoring choke point.

    Pure GATE over a ``HybridDecision``:
      * ``approve`` / ``block`` pass through byte-for-byte (never touched);
      * only a ``challenge`` is eligible;
      * dormant until the user's profile is configured
        (``verification_count >= MIN_VERIFICATIONS_TO_CONFIGURE``);
      * a challenge that matches verified behaviour is CONFIRMED (stays);
      * a challenge matching none of it is escalated one band to ``block``.

    Always returns a ``SecondaryDecision`` (a ``HybridDecision`` subclass, so
    callers that only know the primary shape keep working).
    """
    from core.models import SecondaryBehaviorProfile

    if decision.verdict != "challenge":
        return decision

    profile = SecondaryBehaviorProfile.objects.filter(user=session.user).first()
    count = profile.verification_count if profile is not None else 0
    configured = count >= MIN_VERIFICATIONS_TO_CONFIGURE

    familiarity = None
    action = None
    if profile is not None and profile.verified_contexts:
        familiarity = familiarity_score(session, profile.verified_contexts or [])
        if configured:
            action = secondary_action_for(familiarity)

    reasons = list(decision.triggered_reasons)
    verdict = decision.verdict
    if action == "escalate":
        reasons.append({"code": "secondary_behavior_escalation", "weight": 0})
        verdict = "block"
    elif action == "confirm":
        reasons.append({"code": "secondary_behavior_confirm", "weight": 0})

    return SecondaryDecision(
        session_id=decision.session_id,
        score=decision.score,
        verdict=verdict,
        triggered_reasons=reasons,
        ml_probability=getattr(decision, "ml_probability", 0.0),
        context_override_applied=getattr(
            decision, "context_override_applied", False
        ),
        keystroke_override_applied=getattr(
            decision, "keystroke_override_applied", False
        ),
        secondary_profile_configured=configured,
        secondary_verification_count=count,
        secondary_familiarity=familiarity,
        secondary_action=action,
    )
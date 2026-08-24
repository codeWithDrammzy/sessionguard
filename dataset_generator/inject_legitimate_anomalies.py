"""
Legitimate-anomaly injector for SessionGuard
============================================

Adds a small set of sessions that LOOK suspicious on the surface but carry
ground truth ``is_legitimate_anomaly=True, is_attack=False`` -- honest
customers whose behaviour superficially resembles takeover patterns.

WHY THIS EXISTS
---------------
The challenge brief explicitly warns that neither a shared family phone
nor a genuine post-loss SIM replacement is fraud. A detector trained/
tuned only on baseline + attacks would learn "any deviation = fraud" and
punish exactly those customers. These rows are the calibration evidence
that thresholds/features can separate *anomalous-but-honest* from *fraud*
-- they exist to keep false positives measurable and honest.

TWO CATEGORIES
--------------
* ``family_shared_phone`` (6 cases, app-capable victims): a relative uses
  the SAME physical phone/SIM/location but at THEIR preferred hour,
  transfers unusual-for-the-owner amounts, to THEIR own contacts. Nothing
  about the credentials is new (is_new_device=False, is_new_sim=False) --
  crucially, device+SIM+location never change together, so this pattern
  must NOT trip the combined_device_location_flag invariant. Only timing,
  amount and recipient differ.
* ``genuine_sim_swap`` (4 cases, any channel): customer really lost their
  phone -> genuinely NEW SIM (and, on app, a new handset), but still in
  their normal area, at their normal hours, doing ordinary transactions
  to people they normally pay. The SIM change here is REAL LIFE, not
  attack -- the exact scenario naive SIM-change rules punish wrongly.

DESIGN DECISIONS WORTH AUDITING
-------------------------------
* Seed 45: FOURTH independent RNG stream (42 users, 43 sessions,
  44 attacks) so datasets cannot silently correlate.
* Anomaly victims are excluded from attack victims (and vice versa): a
  user playing both roles would muddy per-user evaluation. Existing
  attack targets are read from stored FraudLabel rows -- same
  derive-from-database principle as everything else.
* ``VictimProfile`` and ``pick_attack_time`` are IMPORTED from
  inject_attacks.py rather than duplicated: anomalies must contrast
  against genuine stored history, and helper reuse keeps token formats
  (fresh hex SIMs/fingerprints, timing resolution) identical across
  generators.
* ``BehavioralFeatures`` are NOT written here -- features come later from
  the shared feature engine over baseline + attacks + anomalies together,
  guaranteeing ONE computation method for the entire dataset.

Usage:
    python dataset_generator/inject_legitimate_anomalies.py
"""

import os
import random
import sys
from decimal import Decimal

# --- Django bootstrap (same pattern as sibling generators) ------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sessionguard_project.settings")

import django  # noqa: E402

django.setup()

# Sibling-module imports resolve because the script directory is sys.path[0]
# when run as a plain script. inject_attacks' __main__ guard prevents its
# main() from executing on import.
from inject_attacks import (  # noqa: E402
    _fresh,
    _hex_string,
    _round_to_step,
    deterministic_uuid,
    pick_attack_time,
    VictimProfile,
)

from django.utils import timezone  # noqa: E402

from core.models import BankUser, FraudLabel, Session, Transaction  # noqa: E402

SEED = 45  # fourth independent stream
FAMILY_CASES = 6  # category A
SIMSWAP_CASES = 4  # category B


def _co_observed_network(rng, profile, geohash, channel):
    """A network identifier genuinely observed WITH this geohash in the
    victim's history, shaped for the channel (IP-like for app, tower for
    USSD) so the reused location stays internally coherent."""
    observed = sorted(profile.networks_per_geo[geohash])
    matching = [n for n in observed if ("." in n) == (channel == "app")]
    return rng.choice(matching or observed)


def build_family_session(rng, profile, now):
    """
    CATEGORY A -- relative borrows the family phone.

    Every CREDENTIAL is deliberately unchanged (same device, same SIM,
    home location): the whole point is that physical-context flags stay
    silent while behavioural context (hour, amount, beneficiary) shifts.
    This is the case the combined_device_location_flag invariant exists
    to protect from false positives.
    """
    user = profile.user

    # Their usual device + usual SIM + usual location: nothing new.
    device = rng.choice(sorted(profile.devices))
    geohash = rng.choice(sorted(profile.networks_per_geo))
    network_id = _co_observed_network(rng, profile, geohash, "app")

    # Outside-hours login: the RELATIVE's schedule, not the owner's.
    timestamp = pick_attack_time(
        rng, now, user.typical_login_hours, want_inside=False
    )

    # A different spender at the same handset: usually well BELOW the
    # owner's floor (smaller day-to-day transactions); occasionally modest
    # new territory above their ceiling -- realistic, never absurd.
    tmin = float(user.typical_transfer_min)
    tmax = float(user.typical_transfer_max)
    if rng.random() < 0.75:
        amount = _round_to_step(tmin * rng.uniform(0.30, 0.70), 50)
    else:
        amount = _round_to_step(tmax * rng.uniform(1.10, 1.35), 50)

    session = Session(
        session_id=deterministic_uuid(rng),
        user=user,
        channel="app",
        timestamp=timestamp,
        device_fingerprint=device,        # SAME phone
        sim_id=profile.real_sim,          # SAME SIM
        location_geohash=geohash,         # home
        ip_or_cell_tower_id=network_id,   # home network
        is_new_device=False,              # nothing physical is new...
        is_new_sim=False,                 # ...so context flags stay quiet
        session_duration_seconds=rng.randint(40, 150),
    )
    tx = Transaction(
        transaction_id=deterministic_uuid(rng),
        session=session,
        timestamp=session.timestamp,
        amount=Decimal(str(max(50, amount))),
        # The relative pays THEIR own contacts, unknown to the owner --
        # a new recipient alone must never equal fraud.
        recipient_id="BNF-" + "".join(
            rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=10)
        ),
        is_new_recipient=True,
        time_since_last_transaction_seconds=None,  # engine derives velocity
    )
    label = FraudLabel(
        session=session,
        is_attack=False,
        attack_type=None,
        is_legitimate_anomaly=True,
    )
    return session, tx, label


def build_simswap_session(rng, profile, now):
    """
    CATEGORY B -- genuine SIM swap after real phone loss.

    Hardware genuinely changed: new SIM always, plus a new handset on the
    app channel (losing a phone means losing both). Everything BEHAVIOURAL
    stayed the same: normal area, normal hours, ordinary amounts, mostly
    familiar beneficiaries. Real life mimicking attack signals -- the
    false-positive acid test.
    """
    user = profile.user

    # Channel follows the customer's real habit; "both" users flip.
    if user.channel_preference == "both":
        channel = rng.choice(["app", "ussd"])
    else:
        channel = user.channel_preference

    sim_id = _fresh(rng, "0123456789abcdef", 64, {profile.real_sim})
    fingerprint = (
        None if channel == "ussd"
        else _fresh(rng, "0123456789abcdef", 64, profile.devices)
    )

    # Still in their normal area (they replaced hardware, not address).
    geohash = rng.choice(sorted(profile.networks_per_geo))
    network_id = _co_observed_network(rng, profile, geohash, channel)

    # Routine unchanged: inside their normal login hours.
    timestamp = pick_attack_time(
        rng, now, user.typical_login_hours, want_inside=True
    )

    # Ordinary transaction: squarely inside their typical range.
    tmin = float(user.typical_transfer_min)
    tmax = float(user.typical_transfer_max)
    amount = _round_to_step(rng.uniform(tmin, tmax), 50)

    # Mostly paying their normal people; occasionally someone new.
    known_recip = rng.random() < 0.70
    recipient_id = (
        rng.choice(user.typical_recipients)
        if known_recip
        else "BNF-" + "".join(
            rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=10)
        )
    )

    session = Session(
        session_id=deterministic_uuid(rng),
        user=user,
        channel=channel,
        timestamp=timestamp,
        device_fingerprint=fingerprint,
        sim_id=sim_id,
        location_geohash=geohash,
        ip_or_cell_tower_id=network_id,
        # New SIM is real; new DEVICE only exists/counts on the app channel
        # (USSD exposes none, per schema).
        is_new_device=(channel == "app"),
        is_new_sim=True,
        session_duration_seconds=(
            rng.randint(40, 160) if channel == "app" else rng.randint(20, 90)
        ),
    )
    tx = Transaction(
        transaction_id=deterministic_uuid(rng),
        session=session,
        timestamp=session.timestamp,
        amount=Decimal(str(amount)),
        recipient_id=recipient_id,
        is_new_recipient=not known_recip,
        time_since_last_transaction_seconds=None,
    )
    label = FraudLabel(
        session=session,
        is_attack=False,
        attack_type=None,
        is_legitimate_anomaly=True,
    )
    return session, tx, label


def main():
    rng = random.Random(SEED)
    now = timezone.now()

    stale = Session.objects.filter(fraud_label__is_legitimate_anomaly=True)
    if stale.count():
        print(
            f"WARNING: deleting {stale.count()} previously injected "
            f"legitimate-anomaly session(s) (cascades to their "
            f"transactions/fraud-labels). Baseline and attack sessions "
            f"are NOT touched."
        )
        stale.delete()

    # Users already carrying attack labels are off-limits: nobody plays
    # both victim and innocent-anomaly in the evaluation set.
    attack_user_ids = set(
        FraudLabel.objects.filter(is_attack=True).values_list(
            "session__user_id", flat=True
        )
    )
    everyone = list(BankUser.objects.exclude(user_id__in=attack_user_ids))
    app_capable = [
        u for u in everyone if u.channel_preference in ("app", "both")
    ]

    if len(app_capable) < FAMILY_CASES or len(everyone) < (
        FAMILY_CASES + SIMSWAP_CASES
    ):
        raise SystemExit("Not enough eligible users; rerun earlier generators")

    family_targets = rng.sample(app_capable, FAMILY_CASES)
    remaining = [u for u in everyone if u not in family_targets]
    simswap_targets = rng.sample(remaining, SIMSWAP_CASES)

    sessions, transactions, labels = [], [], []
    for user in family_targets:
        s, tx, lbl = build_family_session(rng, VictimProfile(user), now)
        sessions.append(s); transactions.append(tx); labels.append(lbl)

    for user in simswap_targets:
        s, tx, lbl = build_simswap_session(rng, VictimProfile(user), now)
        sessions.append(s); transactions.append(tx); labels.append(lbl)

    Session.objects.bulk_create(sessions)
    Transaction.objects.bulk_create(transactions)
    FraudLabel.objects.bulk_create(labels)

    print_summary(sessions)


def print_summary(sessions):
    family = [s for s in sessions if not s.is_new_sim]
    simswap = [s for s in sessions if s.is_new_sim]

    unchanged = sum(
        1 for s in family if not s.is_new_device and not s.is_new_sim
    )
    frac = 100.0 * unchanged / len(family) if family else 0.0

    print()
    print("=" * 62)
    print(f"Legitimate-anomaly injection summary (seed={SEED})")
    print("=" * 62)
    print(f"Anomaly sessions created      : {len(sessions)}")
    print(f"  family_shared_phone (A)     : {len(family)}")
    print(f"  genuine_sim_swap (B)        : {len(simswap)}")
    print()
    print("Category A credential purity  : "
          f"{unchanged}/{len(family)} kept device+SIM unchanged ({frac:.0f}%)")
    print("  -> device & location never change together, so the")
    print("     combined_device_location_flag MUST stay False for these.")
    print("=" * 62)


if __name__ == "__main__":
    main()

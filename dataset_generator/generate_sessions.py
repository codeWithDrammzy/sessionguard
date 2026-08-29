"""
Synthetic baseline session/transaction history generator for SessionGuard
=========================================================================

Generates NORMAL (non-fraudulent) login-session and transfer history for
every existing ``BankUser`` -- the 250 synthetic customers created by
``generate_users.py``. This history IS each user's behavioural baseline:
the reference against which future "scoring-time" sessions and injected
attack scenarios will look anomalous.

ETHICS / DATA PROVENANCE
-----------------------
Entirely synthetic. No real subscribers, IMSIs, IPs, towers or addresses.
Cell-tower IDs and IPs are random tokens (IPs biased toward AFRINIC-looking
ranges purely for cosmetic realism). Geohashes ENCODE the coordinates of
real Nigerian city centres (home/work drawn a few km apart within ONE
assigned city), which is what makes the impossible-travel signal meaningful
downstream -- but no residential address is ever represented: the cells are
1-2 km wide around a city point, and a latitude/longitude jitter is no
address.

REPRODUCIBILITY
---------------
All randomness flows through ONE ``random.Random(43)`` instance -- a
deliberately DIFFERENT seed from the user generator's ``Random(42)``, so
the two datasets are independent random streams and cannot silently
correlate (documented decision per the brief). Structure and distributions
are fully reproducible; only the wall-clock anchoring differs (the window
always ends "today", so absolute dates move with the run date by design).

DETERMINISM NOTES
-----------------
* Session/transaction UUIDs derive from the seeded RNG (never ``uuid4``).
* ``sim_id`` is generated once per user here and tracked in memory only --
  the BankUser model intentionally has no sim column; the SIM identity is
  a property of the user's *history*, materialised per-session.
* ``BehavioralFeatures`` rows are deliberately NOT created for baseline
  sessions: features are computed by the real-time scoring path when a
  session arrives, not retro-fitted onto training history. Any stale
  feature rows from earlier runs are deleted during cleanup.

Usage:
    python dataset_generator/generate_sessions.py
"""

import os
import random
import string
import sys
import uuid
from collections import Counter
from datetime import timedelta
from decimal import Decimal

# --- Django bootstrap (same pattern as generate_users.py) -------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sessionguard_project.settings")

import django  # noqa: E402

django.setup()

from django.utils import timezone  # noqa: E402

from core.geohash_util import NIGERIAN_CITIES, geohash_encode  # noqa: E402
from core.models import (  # noqa: E402
    BankUser,
    BehavioralFeatures,
    KeystrokeDynamics,
    Session,
    Transaction,
)

SEED = 43  # independent stream from generate_users.py's seed 42
HISTORY_WINDOW_DAYS = 21  # three weeks of baseline behaviour

_HEX = "0123456789abcdef"
_RECIPIENT_ALPHABET = string.ascii_uppercase + string.digits


def deterministic_uuid(rng):
    """UUIDv4-format PK derived from the seeded RNG."""
    return uuid.UUID(int=rng.getrandbits(128), version=4)


def _hex_string(rng, length):
    return "".join(rng.choices(_HEX, k=length))


# ---------------------------------------------------------------------------
# Per-user context objects (tracked in memory, never stored on BankUser)
# ---------------------------------------------------------------------------

class UserContext:
    """
    Per-customer simulation state that has no natural home on the model:
    their one real SIM, and their two familiar network locations.

    Locations: index 0 is "home" (the dominant pattern, ~90% of sessions),
    index 1 is "work" (the ~10% routine variation). Encoding routine
    variation HERE is what later makes attack injection meaningful: a new
    location on an attack session contrasts against a history that was
    *mostly* stable but not robotically so.
    """

    def __init__(self, rng, user):
        self.user = user
        self.sim_id = _hex_string(rng, 64)
        # Home AND work are anchored to the user's ONE assigned city (drawn
        # once). Two locations a few km apart inside the same city model
        # routine commuting; the gap stays far below the impossible-travel
        # distance floor, so routine home/work switching can never trip it.
        city = rng.choice(NIGERIAN_CITIES)
        self.locations = [
            self._make_location(rng, city),
            self._make_location(rng, city),
        ]

        self.last_transaction_dt = None  # for velocity-style gap computation

        # Per-user keystroke baseline: consistent typing rhythm across sessions,
        # with per-session jitter to model natural variation.
        self.keystroke_cpm = rng.uniform(150.0, 350.0)  # chars per minute
        self.keystroke_hold_ms = rng.uniform(80.0, 200.0)  # key hold duration
        self.keystroke_interval_ms = rng.uniform(100.0, 300.0)  # inter-key gap

    @staticmethod
    def _make_location(rng, city):
        # Jitter home/work around the city centre by ~5km (neighbourhood
        # scale), encoded into a precision-6 geohash cell.
        lat = city[1] + rng.uniform(-0.05, 0.05)
        lon = city[2] + rng.uniform(-0.05, 0.05)
        return {
            # App traffic shows an IP; USSD shows a tower ID. Both are kept
            # per location so channel choice doesn't change geography.
            "ip": "%d.%d.%d.%d" % (
                rng.choice([41, 105, 197]),  # AFRINIC-looking first octet
                rng.randint(1, 254),
                rng.randint(1, 254),
                rng.randint(1, 254),
            ),
            "tower": "TWR-" + _hex_string(rng, 8),
            "geohash": geohash_encode(lat, lon),
        }


# ---------------------------------------------------------------------------
# Session generation
# ---------------------------------------------------------------------------

def generate_session_datetimes(rng, user, now):
    """
    Return this user's baseline session times: a Poisson-style bursty
    process at 2-4 sessions/week over the 21-day window.

    Reasoning: real usage is NOT evenly spaced -- people bank in bursts
    (payday clusters, quiet weekends). We draw a personal weekly rate
    U(2,4), then exponential inter-arrival gaps (a homogeneous Poisson
    process), which yields exactly that burstiness. Each raw timestamp is
    then SNAPPED onto one of the user's typical_login_hours windows
    (jittered +/-20 min): baseline history must sit inside declared-normal
    hours, otherwise hour-deviation scoring would flag our own training
    data as anomalous.
    """
    rate_per_week = rng.uniform(2.0, 4.0)
    rate_per_second = rate_per_week / (7 * 24 * 3600)

    start = now - timedelta(days=HISTORY_WINDOW_DAYS)
    raw_times = []
    t = start
    while True:
        t += timedelta(seconds=rng.expovariate(rate_per_second))
        if t >= now:
            break
        raw_times.append(t)

    snapped = []
    for t in raw_times:
        day = t.date()
        window = rng.choice(user.typical_login_hours)  # [[s, e], ...]
        lo, hi = window
        hour = rng.randint(lo, hi)
        minute = rng.randint(0, 59)
        moment = timezone.make_aware(
            timezone.datetime(day.year, day.month, day.day, hour, minute)
        )
        # +/-20 minute human jitter (nobody logs in at the exact minute)
        moment += timedelta(minutes=rng.randint(-20, 20))
        if moment.date() != day:
            # Late-evening jitter pushed past midnight; clamp to keep the
            # session on its simulated day.
            moment = moment.replace(hour=23, minute=59, second=0)
        snapped.append(moment)

    snapped.sort()
    return snapped


def build_sessions_for_user(rng, ctx, now):
    """Materialise Session objects (unsaved) plus their chosen locations."""
    user = ctx.user
    sessions = []

    for moment in generate_session_datetimes(rng, user, now):
        # Channel: follow stated preference; "both" users flip ~50/50.
        if user.channel_preference == "both":
            channel = rng.choice(["app", "ussd"])
        else:
            channel = user.channel_preference

        # Routine-vs-varied location, 90/10 (home vs work).
        location = (
            ctx.locations[0] if rng.random() < 0.90 else ctx.locations[1]
        )

        sessions.append(
            Session(
                session_id=deterministic_uuid(rng),
                user=user,
                channel=channel,
                timestamp=moment,
                # Rich device identity only exists on the app channel;
                # feature phones have nothing to fingerprint (schema: NULL).
                device_fingerprint=(
                    rng.choice(user.registered_devices)
                    if channel == "app"
                    else None
                ),
                sim_id=ctx.sim_id,  # their one real SIM, reused every time
                ip_or_cell_tower_id=(
                    location["ip"] if channel == "app" else location["tower"]
                ),
                location_geohash=location["geohash"],
                # Baseline history contains nothing new BY DEFINITION.
                is_new_device=False,
                is_new_sim=False,
                # USSD menu flows are shorter than app journeys, hence the
                # tighter duration band.
                session_duration_seconds=(
                    rng.randint(30, 180)
                    if channel == "app"
                    else rng.randint(20, 90)
                ),
            )
        )
    return sessions


# ---------------------------------------------------------------------------
# Transaction generation
# ---------------------------------------------------------------------------

def build_transactions_for_user(rng, ctx, sessions):
    """
    Attach 0-2 transactions to each session and compute
    time_since_last_transaction_seconds against the user's true previous
    transfer (NULL for their first-ever one).

    Reasoning on the mix: many sessions are balance checks, so transfers
    are the minority outcome -- weights 45/35/20 for 0/1/2 transfers keep
    transactions special rather than reflexive. Amounts are log-uniform
    inside the user's typical band (people transact nearer their floor
    than their ceiling far more often). Recipients: 85% from the known
    circle, 15% one-off strangers -- ordinary life includes paying someone
    new occasionally, which is precisely why new_recipient_flag alone must
    never be treated as an attack signal downstream.
    """
    user = ctx.user
    tmin = float(user.typical_transfer_min)
    tmax = float(user.typical_transfer_max)

    known = user.typical_recipients
    transactions = []

    import math  # local import keeps module top clean; used only here

    for session in sessions:
        n_transfers = rng.choices([0, 1, 2], weights=[45, 35, 20], k=1)[0]
        for _ in range(n_transfers):
            amount = math.exp(rng.uniform(math.log(tmin), math.log(tmax)))
            amount = int(round(amount / 50) * 50)
            amount = max(50, min(amount, int(tmax)))

            is_new = rng.random() < 0.15
            recipient = (
                "BNF-" + "".join(rng.choices(_RECIPIENT_ALPHABET, k=10))
                if is_new
                else rng.choice(known)
            )

            gap_seconds = (
                int((session.timestamp - ctx.last_transaction_dt).total_seconds())
                if ctx.last_transaction_dt is not None
                else None
            )

            transactions.append(
                Transaction(
                    transaction_id=deterministic_uuid(rng),
                    session=session,
                    timestamp=session.timestamp,  # transfer happens in-session
                    amount=Decimal(str(amount)),
                    recipient_id=recipient,
                    is_new_recipient=is_new,
                    time_since_last_transaction_seconds=gap_seconds,
                )
            )
            ctx.last_transaction_dt = session.timestamp

    return transactions


# ---------------------------------------------------------------------------
# Main + summary
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(SEED)
    now = timezone.now()

    stale = (
        Session.objects.count(),
        Transaction.objects.count(),
        BehavioralFeatures.objects.count(),
    )
    if any(stale):
        print(
            f"WARNING: clearing existing baseline data -- "
            f"{stale[0]} session(s), {stale[1]} transaction(s), "
            f"{stale[2]} behavioral-feature row(s). Deleting Sessions "
            f"cascades to their transactions/features/fraud-labels."
        )
        Session.objects.all().delete()  # cascades: transactions, features, labels

    users = list(BankUser.objects.all())
    if not users:
        raise SystemExit(
            "No BankUser records found -- run generate_users.py first."
        )

    all_sessions, all_transactions, all_keystrokes = [], [], []
    user_contexts = {}
    for user in users:
        ctx = UserContext(rng, user)
        user_contexts[user.user_id] = ctx
        sessions = build_sessions_for_user(rng, ctx, now)
        transactions = build_transactions_for_user(rng, ctx, sessions)
        # Generate KeystrokeDynamics for app sessions using this user's
        # consistent typing rhythm with per-session jitter.
        for session in sessions:
            if session.channel == "app":
                cpm = max(50.0, ctx.keystroke_cpm + rng.gauss(0, 15.0))
                hold = max(30.0, ctx.keystroke_hold_ms + rng.gauss(0, 10.0))
                interval = max(50.0, ctx.keystroke_interval_ms + rng.gauss(0, 12.0))
                all_keystrokes.append(
                    KeystrokeDynamics(
                        session=session,
                        avg_hold_time_ms=hold,
                        avg_interval_ms=interval,
                        typing_speed_cpm=cpm,
                    )
                )
        all_sessions.extend(sessions)
        all_transactions.extend(transactions)

    Session.objects.bulk_create(all_sessions)
    KeystrokeDynamics.objects.bulk_create(all_keystrokes)
    Transaction.objects.bulk_create(all_transactions)

    print_summary(now, users, all_sessions, all_transactions)


def print_summary(now, users, sessions, transactions):
    channels = Counter(s.channel for s in sessions)
    total_sessions = len(sessions)
    total_tx = len(transactions)

    per_user = Counter(s.user_id for s in sessions)
    per_user_counts = list(per_user.values())

    # Count sessions at the secondary ("work") location by re-deriving each
    # user's dominant geohash from the data itself; anything else counts as
    # routine variation.
    dominant = {}
    for s in sessions:
        dominant.setdefault(s.user_id, Counter())[s.location_geohash] += 1
    secondary_share = sum(
        sum(counts.values()) - counts.most_common(1)[0][1]
        for counts in dominant.values()
    )
    secondary_pct = (
        100.0 * secondary_share / total_sessions if total_sessions else 0.0
    )

    new_recips = sum(1 for t in transactions if t.is_new_recipient)
    null_gaps = sum(
        1 for t in transactions if t.time_since_last_transaction_seconds is None
    )
    durations = [s.session_duration_seconds for s in sessions]
    keystroke_count = KeystrokeDynamics.objects.count()

    def pct(n, d):
        return f"{100.0 * n / d:.1f}%" if d else "n/a"

    print()
    print("=" * 62)
    print(f"Baseline history summary (seed={SEED}, window={HISTORY_WINDOW_DAYS}d)")
    print("=" * 62)
    print(f"Users processed               : {len(users)}")
    print(f"Sessions created              : {total_sessions}")
    print(f"  per user                    : min={min(per_user_counts)}  "
          f"max={max(per_user_counts)}  "
          f"avg={total_sessions / len(users):.1f}")
    print(f"  channel split               : app={channels.get('app', 0)}  "
          f"ussd={channels.get('ussd', 0)}")
    print(f"  secondary-location sessions : {secondary_share} ({pct(secondary_share, total_sessions)})")
    print(f"Keystroke dynamics records    : {keystroke_count}")
    print(f"Transactions created          : {total_tx}")
    print(f"  new-recipient transfers     : {new_recips} ({pct(new_recips, total_tx)})")
    print(f"  NULL gap (first-ever tx)    : {null_gaps}")
    print(f"Session duration (s)          : min={min(durations)}  "
          f"max={max(durations)}")
    print("=" * 62)


if __name__ == "__main__":
    main()

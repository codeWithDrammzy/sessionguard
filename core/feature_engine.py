"""
SessionGuard shared feature engine
==================================

ONE implementation of the feature computation used by BOTH:
  * batch processing of the full synthetic dataset (training/evaluation), and
  * live scoring in the future real-time API.

Sharing this code is not a convenience -- it is a correctness requirement.
If training features and inference features were computed by different
logic (even subtly different thresholds or history windows), the model
would learn patterns that do not exist at scoring time. There is exactly
one ``compute_features()``; every consumer calls it.

CAUSALITY GUARANTEE
-------------------
Every field is computed from the session's own data plus that user's PRIOR
history only (sessions with strictly earlier timestamps). No future data is
ever consulted -- a live system could not have it at scoring time. In batch
mode this is guaranteed by processing sessions ordered per user by
timestamp and updating each user's history tracker AFTER computing their
session's features.

DESIGN NOTES
------------
* ``compute_features(session)`` returns an UNSAVED BehavioralFeatures row:
  callers decide persistence (batch uses bulk_create for speed; a live API
  would save() immediately).
* With no explicit history passed, the function lazily loads that user's
  priors from the DB -- the right behaviour for single-session scoring.
* The combined_device_location_flag is NOT re-implemented here: it
  delegates to ``resolve_combined_device_location_flag`` in core.models --
  the same function BehavioralFeatures.save() uses. One invariant, one
  definition (bulk_create bypasses save(), so save()-only enforcement is
  not enough for batch mode).

KNOWN LIMITATION (for the write-up's honest-failure-analysis section):
  menu_timing_deviation_score needs >= 3 prior USSD sessions to form a
  duration baseline; users with less history get a default 0.0 ("cannot
  judge yet"), meaning very new USSD customers are temporarily blind to
  this particular signal rather than falsely flagged by it.

Usage:
    python core/feature_engine.py            # batch over whole dataset
    from core.feature_engine import compute_features   # live scoring path
"""

import os
import sys
from collections import Counter, defaultdict, deque
from datetime import timedelta
from statistics import mean, pstdev

# --- Django bootstrap ONLY when run directly --------------------------------
# Mirrors the dataset_generator scripts. When imported by Django code
# (views, management commands), the caller's project is already configured
# and this block is skipped entirely.
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE", "sessionguard_project.settings"
    )
    import django

    django.setup()

from core.models import (  # noqa: E402
    BehavioralFeatures,
    FraudLabel,
    Session,
    resolve_combined_device_location_flag,
)

BATCH_SIZE = 500  # bulk_create chunk size
VELOCITY_WINDOW = timedelta(minutes=5)
MIN_USSD_BASELINE_SESSIONS = 3  # below this: not enough history to judge
HOUR_CAP_MINUTES = 360  # >=6h outside any window => hour_deviation 1.0


class UserHistory:
    """
    Incremental, strictly-causal view of ONE user's prior sessions.

    Fed one session at a time (in chronological order); every query answers
    "what had the system seen BEFORE this session?" Nothing here ever looks
    ahead. The same class serves live scoring: replay the user's past
    sessions into it, then score the incoming one.
    """

    def __init__(self):
        self.device_counts = Counter()
        self.sim_counts = Counter()
        self.geohashes = set()
        self.ussd_durations = []
        # Timestamps kept for velocity; pruned from the left as time moves.
        self.recent_timestamps = deque()

    @property
    def has_prior(self):
        return (
            bool(self.device_counts)
            or bool(self.sim_counts)
            or bool(self.geohashes)
        )

    def dominant_sim(self):
        """Most common SIM seen so far (None if no history)."""
        return self.sim_counts.most_common(1)[0][0] if self.sim_counts else None

    def observe(self, session):
        """Fold a session INTO history. Call only AFTER scoring it."""
        if session.device_fingerprint:
            self.device_counts[session.device_fingerprint] += 1
        self.sim_counts[session.sim_id] += 1
        self.geohashes.add(session.location_geohash)
        if session.channel == "ussd" and session.session_duration_seconds:
            self.ussd_durations.append(session.session_duration_seconds)
        self.recent_timestamps.append(session.timestamp)


def _minute_distance_circular(a, b):
    """Distance between two minute-of-day values on a 24h clock -- 00:00 is
    1 minute from 23:59, not 24 hours. Without this, post-midnight logins
    near a late-evening window would be absurdly over-penalised."""
    d = abs(a - b) % 1440
    return min(d, 1440 - d)


def _hour_deviation(typical_windows, timestamp):
    """
    0.0 inside any window; otherwise distance in hours to the nearest
    window edge (midnight-aware), linearly scaled so >=6h outside
    saturates at 1.0. Uses BankUser.typical_login_hours directly -- it
    exists even for users with zero session history.
    """
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    best = None
    for lo, hi in typical_windows:
        start, end = lo * 60, hi * 60 + 59  # window end-hour inclusive
        if start <= minute_of_day <= end:
            return 0.0
        d = min(
            _minute_distance_circular(minute_of_day, start),
            _minute_distance_circular(minute_of_day, end),
        )
        best = d if best is None else min(best, d)
    return min(1.0, best / HOUR_CAP_MINUTES)


def _amount_deviation(amount, tmin, tmax):
    """
    0.0 within [tmin, tmax]; else proportional overshoot past the violated
    boundary, capped at 1.0 once the amount reaches 2x the boundary
    (symmetric treatment for below-minimum amounts).
    """
    amount = float(amount)
    if tmin <= amount <= tmax:
        return 0.0
    if amount > tmax > 0:
        return min(1.0, (amount - tmax) / tmax)
    if tmin > 0:
        return min(1.0, (tmin - amount) / tmin)
    return 1.0  # degenerate range; treat as maximal deviation


def compute_features(session, history=None):
    """
    Compute the BehavioralFeatures vector for ONE session.

    Returns an UNSAVED instance (caller persists). ``history`` lets batch
    mode supply an incrementally-maintained UserHistory; when omitted, the
    user's prior sessions are loaded from the DB -- the live-scoring shape.
    """
    if history is None:
        history = load_user_history(session.user_id, before=session.timestamp)

    # --- Transactions on this session (may be none) -------------------------
    transactions = list(session.transactions.all())
    amounts = [t.amount for t in transactions]
    new_recipient = any(t.is_new_recipient for t in transactions)

    # --- Flags vs prior history ---------------------------------------------
    # All three context flags share one semantic: "changed" requires
    # something TO compare against. A first-ever session (or a USSD-only
    # history meeting its first app session) has no comparable baseline,
    # so absence of evidence stays False -- never punished for having no
    # history yet. This mirrors sim_change_flag's documented no-prior
    # default and keeps the three flags symmetric for the model.
    device_change = bool(
        history.device_counts
        and session.device_fingerprint
        and session.device_fingerprint not in history.device_counts
    )
    dominant_sim = history.dominant_sim()
    sim_change = dominant_sim is not None and session.sim_id != dominant_sim
    location_change = (
        session.location_geohash not in history.geohashes
        if history.has_prior
        else False
    )

    # --- Velocity -----------------------------------------------------------
    cutoff = session.timestamp - VELOCITY_WINDOW
    while history.recent_timestamps and history.recent_timestamps[0] < cutoff:
        history.recent_timestamps.popleft()
    velocity = len(history.recent_timestamps)  # others in (t-5m, t]

    # --- Amount deviation: worst offending transaction in the session -------
    amount_score = None
    if amounts:
        tmin = float(session.user.typical_transfer_min or 0)
        tmax = float(session.user.typical_transfer_max or 0)
        amount_score = max(
            _amount_deviation(a, tmin, tmax) for a in amounts
        )

    # --- Menu-timing deviation (USSD only) ----------------------------------
    menu_score = None
    if session.channel == "ussd":
        menu_score = 0.0  # honest default while history is too thin
        durations = history.ussd_durations
        dur = session.session_duration_seconds
        if dur and len(durations) >= MIN_USSD_BASELINE_SESSIONS:
            mu = mean(durations)
            sigma = pstdev(durations)
            if sigma > 0:
                z = (dur - mu) / sigma
                # Within ~1 SD of normal -> 0; scales up beyond that,
                # saturating at 1.0 around +3 SD.
                menu_score = max(0.0, min(1.0, (z - 1.0) / 2.0))
            elif dur > mu:
                menu_score = 1.0  # zero-variance history, longer than ALL

    features = BehavioralFeatures(
        session=session,
        hour_deviation_score=_hour_deviation(
            session.user.typical_login_hours, session.timestamp
        ),
        amount_deviation_score=amount_score,
        device_change_flag=device_change,
        sim_change_flag=sim_change,
        location_change_flag=location_change,
        new_recipient_flag=new_recipient,
        velocity_count_5min=velocity,
        menu_timing_deviation_score=menu_score,
    )
    # Delegate to the model's single-source-of-truth resolver -- identical
    # rule to BehavioralFeatures.save(); NOT a re-implementation.
    features.combined_device_location_flag = (
        resolve_combined_device_location_flag(
            features.device_change_flag,
            features.sim_change_flag,
            features.location_change_flag,
        )
    )
    return features


def load_user_history(user_id, before):
    """Replay one user's prior sessions into a fresh UserHistory."""
    history = UserHistory()
    priors = Session.objects.filter(
        user_id=user_id, timestamp__lt=before
    ).order_by("timestamp")
    for s in priors:
        history.observe(s)
    return history


def _label_map():
    """One query for all labels, keyed by session_id (avoids N joins)."""
    return {
        row["session_id"]: row
        for row in FraudLabel.objects.values(
            "session_id", "is_attack", "attack_type", "is_legitimate_anomaly"
        )
    }


def _category_for(session, labels):
    """baseline | attack:<type> | anomaly:family | anomaly:simswap."""
    label = labels.get(session.session_id)
    if label is None:
        return "baseline"
    if label["is_attack"]:
        return f"attack:{label['attack_type']}"
    if label["is_legitimate_anomaly"]:
        return (
            "anomaly:family_shared_phone"
            if not session.is_new_sim
            else "anomaly:genuine_sim_swap"
        )
    return "other"


def compute_features_for_all_sessions(batch_size=BATCH_SIZE):
    """
    Batch-compute features for EVERY session in the database.

    Sessions are processed ordered by (user, timestamp) so each user's
    UserHistory can be maintained incrementally -- O(N) total instead of a
    per-session history query. Rows are persisted with bulk_create in
    chunks; progress prints every chunk.
    """
    existing = BehavioralFeatures.objects.count()
    if existing:
        print(
            f"WARNING: deleting {existing} existing BehavioralFeatures "
            f"row(s) before recomputing fresh over the full dataset."
        )
        BehavioralFeatures.objects.all().delete()

    histories = defaultdict(UserHistory)
    pending, created_total = [], 0
    processed = 0

    sessions = (
        Session.objects.select_related("user")
        .prefetch_related("transactions")
        .order_by("user_id", "timestamp")
    )

    for session in sessions.iterator(chunk_size=batch_size):
        features = compute_features(session, history=histories[session.user_id])
        histories[session.user_id].observe(session)  # AFTER scoring: causality
        pending.append(features)
        processed += 1

        if len(pending) >= batch_size:
            BehavioralFeatures.objects.bulk_create(
                pending, batch_size=batch_size
            )
            created_total += len(pending)
            print(f"  processed {created_total} sessions...")
            pending = []

    if pending:
        BehavioralFeatures.objects.bulk_create(pending, batch_size=batch_size)
        created_total += len(pending)

    print(f"  processed {created_total} sessions... done.")
    print_validation_summary(created_total)


def print_validation_summary(created_total):
    """
    Sanity check: flag incidence broken down by session category. The key
    expectation: combined_device_location_flag=True concentrated almost
    entirely in credential_theft + obvious sim-swap attacks, near-zero on
    baseline, and ZERO on family_shared_phone anomalies.
    """
    flags = [
        "device_change_flag",
        "sim_change_flag",
        "location_change_flag",
        "combined_device_location_flag",
    ]
    labels = _label_map()
    tally = defaultdict(Counter)
    totals = Counter()

    for f in (
        BehavioralFeatures.objects.select_related("session").iterator()
    ):
        cat = _category_for(f.session, labels)
        totals[cat] += 1
        for flag in flags:
            if getattr(f, flag):
                tally[flag][cat] += 1

    categories = sorted(totals)
    print()
    print("=" * 78)
    print(f"Feature validation summary ({created_total} rows computed)")
    print("=" * 78)
    header = f"{'flag':<32}" + "".join(f"{c[:18]:>20}" for c in categories)
    print(header)
    for flag in flags:
        row = f"{flag:<32}"
        row += "".join(f"{tally[flag].get(c, 0):>20}" for c in categories)
        print(row)
    print("-" * 78)

    family_cat = "anomaly:family_shared_phone"
    bad = tally["combined_device_location_flag"].get(family_cat, 0)
    n_family = totals.get(family_cat, 0)
    verdict = "PASS" if bad == 0 else "FAIL -- INVESTIGATE"
    print(
        f"CRITICAL CHECK: combined_device_location_flag=True among "
        f"family_shared_phone anomalies: {bad}/{n_family} -> {verdict}"
    )
    print("(All six must be False: same phone, same SIM, home location.)")
    print("=" * 78)


def run():
    """Entry point for direct execution."""
    compute_features_for_all_sessions()


if __name__ == "__main__":
    run()

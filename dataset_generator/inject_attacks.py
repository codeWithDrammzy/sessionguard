"""
Attack-scenario injector for SessionGuard
=========================================

Adds ATTACK sessions + transactions on top of the baseline population,
contrasting each attack against THAT SPECIFIC VICTIM'S real stored history
(not a regenerated lookalike). Together with ``generate_sessions.py`` this
produces the labelled evaluation dataset: baseline sessions are implicitly
benign, injected sessions carry a ``FraudLabel(is_attack=True)``.

Two attack archetypes (60/40 split of targets):

* ``credential_thief`` -- loud takeover: new device, new SIM, a leap to a
  far international city (>=3000 km away) a few minutes after the victim's
  last recorded login, plus a near-maximum transfer to a brand-new
  beneficiary. Every classical signal fires at once -- including the
  physically-impossible-travel signal the detector must never miss.
* ``patient_low_and_slow`` -- quiet takeover: NEW DEVICE is the only hard
  signal. SIM unchanged (attacker controls OTP delivery), location and
  login hour blend into the victim's routine, amount deliberately held at
  40-70% of the victim's typical maximum, beneficiary 50/50 familiar.
  Designed to defeat naive threshold rules that key on amount/location/
  timing alone.
* ``sim_swap_takeover`` -- USSD-native takeover for feature-phone victims,
  who have no device fingerprint at all (schema: always NULL on USSD
  sessions). The SWAPPED SIM is unavoidably visible and IS the primary
  fraud signal on this channel -- unlike app-world SIM changes, which are
  routine. Ships in the same obvious/patient duality: the loud variant
  leaps to a far international city minutes after the victim's last
  session, at near-maximum amounts with fumbling 100-180s sessions; the
  patient variant reuses the victim's real location and login hours,
  stays under the amount ceiling, runs slightly-elevated 70-110s sessions,
  and differs mainly in menu-navigation pace (moderate deviation vs the
  victim's rhythm).

DESIGN DECISIONS WORTH AUDITING
-------------------------------
* Seed 44: a THIRD independent RNG stream (users=42, sessions=43) so no
  dataset can silently correlate with another.
* Targets form two DISJOINT pools drawn from the SAME seed-44 stream:
  12% of app-capable customers (channel_preference in {"app", "both"})
  for the device-signal archetypes, then 12% of USSD-only customers for
  ``sim_swap_takeover`` (sampled AFTER the app side so RNG consumption
  order is deterministic). No overlap logic is needed across pools since
  a user belongs to exactly one pool by channel_preference.
* Profiles are DERIVED FROM THE DATABASE at injection time -- known
  devices/SIM/geohashes/network IDs are aggregated from each victim's
  existing Session rows -- so an attack contrasts against the victim's
  genuine baseline, however noisy it is.
* Attack timing: PATIENT attacks resolve the tension between "shortly
  after now" and the inside/outside-window constraints by taking the
  NEAREST compliant moment after scoring time (rolling forward hour-by-
  hour, capped well within a day). LOUD attacks instead anchor 10-40
  minutes after the victim's OWN last baseline session and leap to a far
  city -- the impossible-travel leap IS their discriminating signal, so
  they no longer need an off-hours roll to stand out (see
  ``anchored_attack_time``/``far_city_geohash``).
* Exactly ONE transfer per attack session (the heist itself);
  ``time_since_last_transaction_seconds`` is computed against the victim's
  real last baseline transaction for continuity.
* ``BehavioralFeatures`` are NOT written by this script at all -- they are
  computed LATER by the shared feature engine, identically over baseline +
  attack sessions (same architectural decision as ``generate_sessions.py``).
  Hand-stamping a feature only on attack rows would let a model learn
  "high value = attack" from our generation choices instead of from genuine
  behaviour -- a label-leakage risk that inflates evaluation numbers. The
  honest signal already lives in the data: USSD attack session durations
  (100-180s obvious / 70-110s patient) are genuinely elevated against each
  victim's real historical USSD pattern (20-90s), and the engine will
  derive ``menu_timing_deviation_score`` FROM that difference, the same
  way it does for every other session in the dataset.

Usage:
    python dataset_generator/inject_attacks.py
"""

import os
import random
import string
import sys
import uuid
from collections import Counter, defaultdict
from datetime import timedelta
from decimal import Decimal
from statistics import mean

# --- Django bootstrap (same pattern as sibling generators) ------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sessionguard_project.settings")

import django  # noqa: E402

django.setup()

from django.utils import timezone  # noqa: E402

from core.geohash_util import FAR_ATTACK_CITIES, geohash_encode  # noqa: E402
from core.models import BankUser, FraudLabel, KeystrokeDynamics, Session, Transaction  # noqa: E402

SEED = 44  # third independent stream (42 users, 43 sessions)
TARGET_COUNT = 30  # 12% of the 250-user population, app-capable pool
USSD_TARGET_SHARE = 0.12  # 12% of the USSD-only pool (~98 users -> ~12)
OBVIOUS_SHARE = 0.60  # credential_theft / obvious-sim-swap share of targets

_HEX = "0123456789abcdef"
_RECIPIENT_ALPHABET = string.ascii_uppercase + string.digits


def deterministic_uuid(rng):
    return uuid.UUID(int=rng.getrandbits(128), version=4)


def _hex_string(rng, length=64):
    return "".join(rng.choices(_HEX, k=length))


def _round_to_step(value, step=50):
    return max(step, int(round(value / step)) * step)


def _fresh(rng, alphabet, length, forbidden):
    """Random token guaranteed absent from `forbidden` (collision-impossible
    in practice, but the explicit check documents the intent and guards the
    invariant even if baselines ever contain tiny value spaces)."""
    while True:
        token = "".join(rng.choices(alphabet, k=length))
        if token not in forbidden:
            return token


def _fresh_ip(rng, forbidden):
    while True:
        ip = "%d.%d.%d.%d" % (
            rng.choice([41, 105, 197]),
            rng.randint(1, 254),
            rng.randint(1, 254),
            rng.randint(1, 254),
        )
        if ip not in forbidden:
            return ip


# ---------------------------------------------------------------------------
# Victim profiling: read the REAL baseline out of the database
# ---------------------------------------------------------------------------

class VictimProfile:
    """Aggregated straight from the victim's stored Session rows."""

    def __init__(self, user):
        sessions = list(user.sessions.prefetch_related("keystroke_dynamics").all())
        if not sessions:
            raise ValueError(
                f"user {user.user_id} has no baseline history; "
                f"run generate_sessions.py first"
            )

        # Devices actually seen (app sessions only carry fingerprints).
        self.devices = {
            s.device_fingerprint for s in sessions if s.device_fingerprint
        }
        # Their one real SIM = the dominant value in history.
        self.real_sim = Counter(s.sim_id for s in sessions).most_common(1)[0][0]

        # Geohashes actually used, dominant first, plus the exact
        # network identifiers observed TOGETHER with each geohash -- so a
        # patient attack can reuse a coherent (location, network) pair
        # instead of inventing an inconsistent mashup.
        geo_counter = Counter(s.location_geohash for s in sessions)
        self.geohashes_by_rank = [g for g, _ in geo_counter.most_common()]
        networks_per_geo = defaultdict(set)
        for s in sessions:
            networks_per_geo[s.location_geohash].add(s.ip_or_cell_tower_id)
        self.networks_per_geo = networks_per_geo

        self.last_tx = (
            Transaction.objects.filter(session__user=user)
            .order_by("-timestamp")
            .first()
        )

        # The victim's MOST RECENT baseline session (any channel) -- the
        # anchor point for loud-attack timing: an obvious takeover strikes
        # 10-40 minutes after THIS session, at a city >=3000 km away, which
        # implies a physically impossible travel speed.
        self.last_session = max(sessions, key=lambda s: s.timestamp)

        self.forbidden_geo = set(self.geohashes_by_rank)

        self.user = user

        # Keystroke baseline from app sessions (for generating mismatched
        # attack keystroke dynamics).
        ks_hold_times, ks_intervals, ks_cpm = [], [], []
        for s in sessions:
            ks = s.keystroke_dynamics.first()
            if ks is not None:
                ks_hold_times.append(ks.avg_hold_time_ms)
                ks_intervals.append(ks.avg_interval_ms)
                ks_cpm.append(ks.typing_speed_cpm)
        self.keystroke_hold_mean = mean(ks_hold_times) if ks_hold_times else 150.0
        self.keystroke_interval_mean = mean(ks_intervals) if ks_intervals else 200.0
        self.keystroke_cpm_mean = mean(ks_cpm) if ks_cpm else 250.0


# ---------------------------------------------------------------------------
# Attack construction
# ---------------------------------------------------------------------------

def pick_attack_time(rng, now, windows, want_inside):
    """
    Nearest moment >= now(+jitter) whose HOUR satisfies the window rule.

    want_inside=True  -> hour falls within one of typical_login_hours
                         (patient attack blends into routine timing).
    want_inside=False -> hour falls outside EVERY window (obvious attack
                         strikes at an unusual hour).

    Rolls forward hourly until compliant (<= ~47 steps worst case); the
    initial 10-90min jitter keeps timestamps from being suspiciously
    round-numbered across runs.
    """
    inside = set()
    for lo, hi in windows:
        inside.update(range(lo, hi + 1))
    allowed = inside if want_inside else set(range(24)) - inside
    if not allowed:  # defensive: 24h-covered user (not producible today)
        allowed = {(set(range(24)) - inside) or {0}}

    t = now + timedelta(minutes=rng.randint(10, 90))
    for _ in range(49):
        if t.hour in allowed:
            return t
        t += timedelta(hours=1)
    return t  # unreachable in practice; keeps type-checkers happy


def anchored_attack_time(profile, rng):
    """
    Loud-attack timing: 10-40 minutes after the victim's LAST BASELINE
    session.

    Anchoring to that precise prior session (rather than an arbitrary
    near-now moment) guarantees the impossible-travel judgement has a
    determinable "last location" to compare against: the attack leapfrogs
    from a place the victim just proved they were at, at a speed no human
    can travel. The routine-hour-of-day signal is deliberately NOT used
    for loud attacks anymore -- the leap is the discriminating signal, so
    mixing an off-hours roll into the timing would only blur the model's
    clean separation between "obvious blink-attack" and other archetypes.
    """
    return profile.last_session.timestamp + timedelta(
        minutes=rng.randint(10, 40)
    )


def far_city_geohash(rng):
    """Precision-6 geohash of a random far-international city. Every choice
    is >=3000 km from every Nigerian city, so even the tightest 10-minute
    window implies far beyond the 900 km/h impossible-travel threshold;
    and the anchor is 10-40 minutes, so the flag always fires."""
    _name, lat, lon = rng.choice(FAR_ATTACK_CITIES)
    return geohash_encode(lat, lon)


def build_transaction(rng, profile, session, amount_ratio_range, force_new_recipient):
    """
    One heist transfer on the attack session.

    Amount is drawn as a RATIO of the victim's real typical_transfer_max --
    ratios (not absolute naira) are what make the two archetypes separable
    in the summary: obvious ~1.0x+, patient 0.4-0.7x.
    """
    user = profile.user
    tmax = float(user.typical_transfer_max)
    amount = _round_to_step(tmax * rng.uniform(*amount_ratio_range))
    amount = min(amount, int(tmax * 2))  # hard ceiling for sanity

    if force_new_recipient:
        is_new = True
        recipient = "BNF-" + "".join(
            rng.choices(_RECIPIENT_ALPHABET, k=10)
        )
    else:  # patient archetype: coin-flip familiar vs stranger
        is_new = rng.random() < 0.5
        recipient = (
            "BNF-" + "".join(rng.choices(_RECIPIENT_ALPHABET, k=10))
            if is_new
            else rng.choice(user.typical_recipients)
        )

    gap = None
    if profile.last_tx is not None:
        # Clamp at 0: a negative gap would mean the heist happened BEFORE the
        # victim's last baseline transaction, which either indicates a stale
        # baseline row or a clock/anchoring inconsistency -- never a real
        # credit ("the last transfer is in the future"). Zero is the honest
        # result: "another transfer arrived immediately after".
        gap = int(max(0, (session.timestamp - profile.last_tx.timestamp).total_seconds()))

    return Transaction(
        transaction_id=deterministic_uuid(rng),
        session=session,
        timestamp=session.timestamp,
        amount=Decimal(str(amount)),
        recipient_id=recipient,
        is_new_recipient=is_new,
        time_since_last_transaction_seconds=gap,
    ), amount / tmax if tmax else 0.0


def build_obvious_attack(rng, profile, now):
    """credential_theft: every classical signal fires simultaneously --
    new device, new SIM, and a physically-impossible leap to a far
    international city minutes after the victim's last baseline session."""
    user = profile.user

    session = Session(
        session_id=deterministic_uuid(rng),
        user=user,
        channel="app",
        timestamp=anchored_attack_time(profile, rng),
        device_fingerprint=_fresh(rng, _HEX, 64, profile.devices),
        sim_id=_fresh(rng, _HEX, 64, {profile.real_sim}),
        location_geohash=far_city_geohash(rng),
        ip_or_cell_tower_id=_fresh_ip(rng, set()),
        is_new_device=True,
        is_new_sim=True,
        # Scripted/rushed takeover: far quicker than any human flow.
        session_duration_seconds=rng.randint(12, 45),
    )
    tx, ratio = build_transaction(
        rng, profile, session,
        amount_ratio_range=(0.95, 1.25),  # at/just past the victim's ceiling
        force_new_recipient=True,
    )
    label = FraudLabel(
        session=session,
        is_attack=True,
        attack_type=FraudLabel.ATTACK_CREDENTIAL_THEFT,
        is_legitimate_anomaly=False,
    )
    # Obvious keystroke mismatch: attacker types MUCH faster (scripted/bot),
    # dramatically different from the victim's natural rhythm.
    keystroke = KeystrokeDynamics(
        session=session,
        avg_hold_time_ms=max(20.0, profile.keystroke_hold_mean * rng.uniform(0.3, 0.5)),
        avg_interval_ms=max(30.0, profile.keystroke_interval_mean * rng.uniform(0.2, 0.4)),
        typing_speed_cpm=profile.keystroke_cpm_mean * rng.uniform(2.0, 3.5),
    )
    return session, tx, label, ratio, keystroke


def build_patient_attack(rng, profile, now):
    """
    patient_low_and_slow: the new DEVICE is the only fired signal.

    SIM kept real (attacker holds OTP delivery), location + hour blended
    into the victim's routine, amount under the radar. Only the device
    change betrays the takeover -- exactly the pattern that defeats naive
    amount/location/hour rulesets.

    Keystroke dynamics are subtly different: the attacker observed the
    victim's rhythm but can't replicate it exactly. Hold times and
    intervals are shifted ~15-30% from the victim's mean -- noticeable
    in aggregate but not obviously jarring on any single measurement.
    """
    user = profile.user

    # Reuse one of the victim's REAL geohashes...
    geohash = rng.choice(profile.geohashes_by_rank)
    # ...with a network identifier genuinely observed at that geohash,
    # preferring IP-shaped values since the attack arrives over the app.
    observed = sorted(profile.networks_per_geo[geohash])
    ips = [n for n in observed if "." in n] or observed
    network_id = rng.choice(ips)

    session = Session(
        session_id=deterministic_uuid(rng),
        user=user,
        channel="app",
        timestamp=pick_attack_time(rng, now, user.typical_login_hours, True),
        device_fingerprint=_fresh(rng, _HEX, 64, profile.devices),
        sim_id=profile.real_sim,          # <-- unchanged: blends in
        location_geohash=geohash,         # <-- real location: blends in
        ip_or_cell_tower_id=network_id,   # <-- real network: blends in
        is_new_device=True,
        is_new_sim=False,
        # Careful human-paced session, inside the normal app band.
        session_duration_seconds=rng.randint(60, 160),
    )
    tx, ratio = build_transaction(
        rng, profile, session,
        amount_ratio_range=(0.40, 0.70),  # deliberately below the ceiling
        force_new_recipient=False,        # 50/50 known vs new beneficiary
    )
    label = FraudLabel(
        session=session,
        is_attack=True,
        attack_type=FraudLabel.ATTACK_LOW_AND_SLOW,
        is_legitimate_anomaly=False,
    )
    # Subtle keystroke mismatch: attacker observed the victim's rhythm but
    # can't replicate it exactly. Shifted ~15-30% from victim's mean.
    shift = rng.uniform(0.15, 0.30)
    direction = rng.choice([-1, 1])  # could be faster OR slower
    keystroke = KeystrokeDynamics(
        session=session,
        avg_hold_time_ms=max(
            30.0,
            profile.keystroke_hold_mean * (1.0 + direction * shift)
        ),
        avg_interval_ms=max(
            50.0,
            profile.keystroke_interval_mean * (1.0 + direction * shift)
        ),
        typing_speed_cpm=max(
            80.0,
            profile.keystroke_cpm_mean * (1.0 - direction * shift)
        ),
    )
    return session, tx, label, ratio, keystroke


def build_sim_swap_attack(rng, profile, now, obvious):
    """
    USSD-native takeover for feature-phone victims (no fingerprint exists,
    so fraud must signal through SIM identity + location + timing + pace).

    The swapped SIM is unavoidably flagged (is_new_sim=True) -- on USSD a
    SIM change IS the primary fraud indicator, unlike the app world where
    SIM changes alone are routine (families share phones, people upgrade).
    Obvious and patient variants differ across four axes: location,
    hour-of-day, amount ratio and session pacing.

    Session DURATION is where the menu-navigation signal lives -- and it is
    left as raw behaviour only, never hand-scored. The durations below are
    genuinely elevated against each specific victim's real historical USSD
    sessions (20-90s per generate_sessions.py), and the shared feature
    engine will later compute an honest ``menu_timing_deviation_score``
    from that difference, identically for every session in the dataset.
    Hand-stamping the score here would be label leakage: the model could
    learn our generation choices instead of actual behavioural contrast.

    ``obvious=True``  -> loud: leap to a far city minutes after the
                         victim's last session, 0.95-1.25x amounts,
                         fumbling 100-180s session -- far outside any
                         normal USSD interaction.
    ``obvious=False`` -> patient: victim's REAL geohash+tower, in-hours,
                         0.40-0.70x amounts, slightly-elevated 70-110s
                         session (attacker observed the victim's rhythm
                         before the swap; some such sessions land inside
                         the normal band -- by design).
    """
    user = profile.user

    if obvious:
        # Leap to a far international city, minutes after the victim's last
        # baseline session (physically impossible travel).
        geohash = far_city_geohash(rng)
        known_networks = set().union(*profile.networks_per_geo.values())
        network_id = "TWR-" + _hex_string(rng, 8)
        while network_id in known_networks:
            network_id = "TWR-" + _hex_string(rng, 8)
        duration = rng.randint(100, 180)
        amount_ratio_range = (0.95, 1.25)
        force_new_recipient = True
    else:
        # Blend in geographically: reuse a REAL (geohash, tower) pair that
        # was actually observed together in this victim's baseline.
        geohash = rng.choice(profile.geohashes_by_rank)
        observed = sorted(profile.networks_per_geo[geohash])
        towers = [n for n in observed if "." not in n] or observed
        network_id = rng.choice(towers)
        duration = rng.randint(70, 110)
        amount_ratio_range = (0.40, 0.70)
        force_new_recipient = False

    session = Session(
        session_id=deterministic_uuid(rng),
        user=user,
        channel="ussd",
        timestamp=(
            anchored_attack_time(profile, rng)
            if obvious
            else pick_attack_time(rng, now, user.typical_login_hours, True)
        ),
        # Schema-consistent: USSD sessions NEVER carry a fingerprint. The
        # absence of a device is not evidence of anything on this channel,
        # so is_new_device stays False even though this is an attack.
        device_fingerprint=None,
        sim_id=_fresh(rng, _HEX, 64, {profile.real_sim}),
        location_geohash=geohash,
        ip_or_cell_tower_id=network_id,
        is_new_device=False,
        is_new_sim=True,
        session_duration_seconds=duration,
    )
    tx, ratio = build_transaction(
        rng,
        profile,
        session,
        amount_ratio_range=amount_ratio_range,
        force_new_recipient=force_new_recipient,
    )
    label = FraudLabel(
        session=session,
        is_attack=True,
        attack_type=FraudLabel.ATTACK_SIM_SWAP_TAKEOVER,
        is_legitimate_anomaly=False,
    )
    return session, tx, label, ratio


# ---------------------------------------------------------------------------
# Main + summary
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(SEED)
    # Anchor every attack AFTER the most recent baseline session, not to the
    # wall clock: attacks must always postdate the victim's last stored
    # baseline row. (The baseline snapshot timestamp can legitimately outrun
    # the runtime clock -- e.g. a regenerated dataset -- and anchoring to the
    # wall clock alone then collides attack times with future-dated rows.)
    latest_baseline = Session.objects.order_by("-timestamp").values_list(
        "timestamp", flat=True
    ).first()
    now = timezone.now()
    if latest_baseline is not None and latest_baseline > now:
        print(
            f"NOTE: baseline extends to {latest_baseline.isoformat()} (ahead "
            f"of wall clock {now.isoformat()}); anchoring attacks to baseline."
        )
        now = latest_baseline

    stale = Session.objects.filter(fraud_label__is_attack=True)
    if stale.count():
        print(
            f"WARNING: deleting {stale.count()} previously injected attack "
            f"session(s) (cascades to their transactions, keystroke dynamics, "
            f"features -- including stamped menu-timing rows -- and "
            f"fraud-labels). Baseline history is NOT touched."
        )
        stale.delete()

    # --- App-capable pool: device-signal archetypes ------------------------
    pool = list(BankUser.objects.filter(channel_preference__in=["app", "both"]))
    if len(pool) < TARGET_COUNT:
        raise SystemExit("Not enough app-capable users; rerun generate_users.py")

    targets = rng.sample(pool, TARGET_COUNT)
    rng.shuffle(targets)
    n_obvious = int(round(TARGET_COUNT * OBVIOUS_SHARE))
    obvious_targets, patient_targets = (
        targets[:n_obvious],
        targets[n_obvious:],
    )

    sessions, transactions, labels, keystrokes = [], [], [], []
    stats = {
        "credential_theft (app)": [],
        "patient_low_and_slow (app)": [],
        "sim_swap_takeover (ussd)": [],
    }

    for user in obvious_targets:
        s, tx, lbl, ratio, ks = build_obvious_attack(rng, VictimProfile(user), now)
        sessions.append(s); transactions.append(tx); labels.append(lbl)
        keystrokes.append(ks)
        stats["credential_theft (app)"].append(ratio)

    for user in patient_targets:
        s, tx, lbl, ratio, ks = build_patient_attack(rng, VictimProfile(user), now)
        sessions.append(s); transactions.append(tx); labels.append(lbl)
        keystrokes.append(ks)
        stats["patient_low_and_slow (app)"].append(ratio)

    # --- USSD-only pool: sim_swap_takeover ---------------------------------
    # Same SEED-44 stream, consumed AFTER the app-side draws so the entire
    # pipeline stays one deterministic sequence. Pools are disjoint by
    # channel_preference, so no overlap logic is required.
    ussd_pool = list(BankUser.objects.filter(channel_preference="ussd"))
    n_ussd_targets = int(round(len(ussd_pool) * USSD_TARGET_SHARE))
    if n_ussd_targets == 0:
        raise SystemExit("No USSD-only users found; rerun generate_users.py")
    ussd_targets = rng.sample(ussd_pool, n_ussd_targets)
    rng.shuffle(ussd_targets)
    n_ussd_obvious = int(round(n_ussd_targets * OBVIOUS_SHARE))

    for user in ussd_targets[:n_ussd_obvious]:
        s, tx, lbl, ratio = build_sim_swap_attack(
            rng, VictimProfile(user), now, obvious=True
        )
        sessions.append(s); transactions.append(tx); labels.append(lbl)
        stats["sim_swap_takeover (ussd)"].append(ratio)

    for user in ussd_targets[n_ussd_obvious:]:
        s, tx, lbl, ratio = build_sim_swap_attack(
            rng, VictimProfile(user), now, obvious=False
        )
        sessions.append(s); transactions.append(tx); labels.append(lbl)
        stats["sim_swap_takeover (ussd)"].append(ratio)

    # Idempotent insert: prune EVERY row matching the deterministic session
    # IDs we are about to (re)create, regardless of label state. A prior run
    # that died part-way can leave label-less attack sessions behind (the
    # label-filtered stale delete above cannot see those), and re-inserting
    # the same UUIDs would violate the primary key.
    orphan_ids = [s.session_id for s in sessions]
    if Session.objects.filter(session_id__in=orphan_ids).exists():
        Session.objects.filter(session_id__in=orphan_ids).delete()

    Session.objects.bulk_create(sessions)
    KeystrokeDynamics.objects.bulk_create(keystrokes)
    Transaction.objects.bulk_create(transactions)
    FraudLabel.objects.bulk_create(labels)

    print_summary(
        stats, obvious_targets, patient_targets, n_ussd_targets
    )


def print_summary(stats, obvious_targets, patient_targets, n_ussd_targets):
    overlap = {u.user_id for u in obvious_targets} & {
        u.user_id for u in patient_targets
    }
    total_attacks = sum(len(v) for v in stats.values())

    print()
    print("=" * 62)
    print(f"Attack injection summary (seed={SEED})")
    print("=" * 62)
    print(f"Targets selected              : app-capable="
          f"{len(obvious_targets) + len(patient_targets)}  "
          f"ussd-only={n_ussd_targets}  (disjoint pools)")
    print(f"Attack sessions created       : {total_attacks}")
    print(f"  credential_theft (app)      : {len(stats['credential_theft (app)'])}")
    print(f"  patient_low_and_slow (app)  : "
          f"{len(stats['patient_low_and_slow (app)'])}")
    print(f"  sim_swap_takeover (ussd)    : "
          f"{len(stats['sim_swap_takeover (ussd)'])}")
    print(f"Keystroke dynamics records    : {KeystrokeDynamics.objects.count()}")
    print(f"All attacks carried a transaction: yes (1 each)")
    print()
    print("Amount vs victim's typical_transfer_max:")
    for kind, ratios in stats.items():
        print(f"  {kind:<28} avg ratio = {sum(ratios)/len(ratios):.2f}x  "
              f"(range {min(ratios):.2f}x-{max(ratios):.2f}x)")
    print()
    print("Note: no BehavioralFeatures rows are written here; menu-timing and")
    print("all other features are computed later by the shared feature")
    print("engine, identically over baseline + attack sessions.")
    print()
    print(f"Victim overlap across types  : {len(overlap)} (must be 0)")
    print("=" * 62)


if __name__ == "__main__":
    main()

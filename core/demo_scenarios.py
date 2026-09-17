"""
Preset demo scenarios for the live Control Room presentation.

Each preset is a COMPLETE payload matching SessionEventSerializer's shape,
built from REAL rows already in the database -- so clicking a button hits
/api/session-event/ (or /api/ussd-event/) exactly like a real bank would,
with no mocked responses anywhere.

The five presets mirror the five behavioural archetypes the pipeline was
built around:

  normal_login     -- genuine baseline identity + habits  -> expect APPROVE
  obvious_attack   -- USSD-native takeover: new SIM/tower/location, huge
                      amount, unknown recipient            -> expect BLOCK
                      (channel choice documented below)
  patient_attack   -- quiet subtle changes only (new device + a typing
                      rhythm that does not match this customer's), everything
                      else inside normal range -> expect CHALLENGE
  family_sharing   -- identical hardware/location, tiny amount to someone
                      new (the family_shared_phone anomaly) -> expect APPROVE
  genuine_simswap  -- new device AND SIM at home, normal habits (the real
                      post-loss recovery case)             -> expect CHALLENGE

USSD representation -- deliberate choice: ``obvious_attack`` runs as USSD.
Rationale: the brief positions USSD as the differentiating channel for
Nigerian banking and sim-swap takeover is its signature attack; showing
the block verdict on the USSD path (and rendering it as a green-screen
terminal in the UI) demonstrates both in one click.
"""

import os
import sys

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE", "sessionguard_project.settings"
    )
    import django

    django.setup()

from decimal import Decimal  # noqa: E402

from collections import Counter  # noqa: E402

from django.utils import timezone  # noqa: E402

from core.feature_engine import (  # noqa: E402
    KEYSTROKE_FULL_CONFIDENCE_SESSIONS,
    MIN_KEYSTROKE_BASELINE_SESSIONS,
    _keystroke_is_plausible,
)
from core.models import BankUser, Session  # noqa: E402


def _baseline_anchor(user, channel):
    """Most recent PRE-TODAY session for this user+channel.

    Excluding today's rows means API/smoke-test debris can never poison a
    demo preset (same defensive pattern as smoke_test_api.py).
    """
    start_today = timezone.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (Session.objects
            .filter(user=user, channel=channel,
                    device_fingerprint__isnull=False,
                    timestamp__lt=start_today)
            .order_by("-timestamp")
            .first())


def _first_user(channel_preference):
    """Prefer a user whose login window covers NOW, so presets behave as
    labelled regardless of presentation time (e.g. genuine_simswap must
    stay inside its override-eligible context); deterministic fallback."""
    hour = timezone.now().hour
    users = BankUser.objects.filter(
        channel_preference=channel_preference
    ).order_by("user_id")
    for u in users:
        if any(s <= hour < e for s, e in u.typical_login_hours):
            return u
    return users.first()


def _is_identity_stable(user, start_today) -> bool:
    """True when the user's latest pre-today session reflects their
    DOMINANT historical SIM (no life-event noise), so 'same SIM as always'
    presets compare against a trustworthy baseline."""
    sims = Counter(
        Session.objects.filter(user=user, timestamp__lt=start_today)
        .values_list("sim_id", flat=True)
    )
    if not sims:
        return False
    dominant = Counter(sims).most_common(1)[0][0]
    anchor = (
        Session.objects.filter(user=user, timestamp__lt=start_today,
                               device_fingerprint__isnull=False)
        .order_by("-timestamp").first()
    )
    return anchor is not None and anchor.sim_id == dominant


def _usable_for_demo(user):
    """A preset must build a real transfer for this user, so only users
    with at least one known recipient and a valid amount band qualify.

    Bank-app signups (bank_views.py) create accounts with
    ``typical_recipients=[]``; such a user must never be selected or the
    preset builder crashes indexing ``typical_recipients[0]``. Filtering
    HERE -- at the selection source -- makes every consumer safe without
    per-callsite guards.
    """
    return bool(user.typical_recipients) and (
        bool(user.typical_transfer_min) and bool(user.typical_transfer_max)
    )


def _stable_users(channel_preference, need):
    """Up to ``need`` DISTINCT demo-ready users, identity-stable ones
    first (awake preferred inside both groups). Distinctness matters:
    every preset gets its OWN user so clicking several scenarios in a
    row can never let one click's persisted event contaminate another's
    behavioural baseline. Users without a usable recipient/amount band
    are filtered out up front (see ``_usable_for_demo``)."""
    hour = timezone.now().hour
    start_today = timezone.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    users = [u for u in BankUser.objects.filter(
        channel_preference=channel_preference).order_by("user_id")
        if _usable_for_demo(u)]

    def awake(u):
        return any(s <= hour < e for s, e in u.typical_login_hours)

    stable_awake = [u for u in users if awake(u)
                    and _is_identity_stable(u, start_today)]
    stable_any = [u for u in users
                  if _is_identity_stable(u, start_today)]
    ordered, seen = [], set()
    for pool in (stable_awake, stable_any, users):
        for u in pool:
            if u.user_id not in seen:
                seen.add(u.user_id)
                ordered.append(u)
            if len(ordered) >= need:
                return ordered
    return ordered


def _best_keystroke_user(excluded_ids):
    """App user with the RICHEST prior keystroke baseline among those not
    already claimed by another preset.

    ``keystroke_deviation`` needs >= MIN_KEYSTROKE_BASELINE_SESSIONS prior
    app sessions with keystroke data before it can fire, and its confidence
    only reaches full strength around KEYSTROKE_FULL_CONFIDENCE_SESSIONS --
    so preferring the deepest history makes the signal land loudly in the
    live demo instead of half-credibility noise. ``excluded_ids`` keeps the
    four app presets on DISTINCT users (a preset must never piggyback on a
    baseline another button's click already polluted).
    """
    start_today = timezone.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    candidates = []
    for u in BankUser.objects.filter(channel_preference="app").order_by("user_id"):
        if u.user_id in excluded_ids:
            continue
        # The preset payload needs a recipient + amount band; skip users the
        # scenario could not actually be built for.
        if not _usable_for_demo(u):
            continue
        n = Session.objects.filter(
            user=u,
            channel="app",
            timestamp__lt=start_today,
            keystroke_dynamics__isnull=False,
        ).count()
        if n >= MIN_KEYSTROKE_BASELINE_SESSIONS:
            candidates.append((n, u))
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1] if candidates else None


def _keystroke_profile(user):
    """Mean typing rhythm across the user's PRIOR app sessions.

    This is exactly the baseline a live scorer has already built for this
    customer (same pre-today window as ``_baseline_anchor``), so the preset
    can synthesize a plausible attacker rhythm from REAL rows -- the
    "built from live data, no mocked responses" contract.
    """
    start_today = timezone.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    holds, intervals, cpm = [], [], []
    hold_stds, interval_stds, pauses, backspaces = [], [], [], []
    priors = (
        Session.objects.filter(
            user=user,
            channel="app",
            timestamp__lt=start_today,
            keystroke_dynamics__isnull=False,
        )
        .prefetch_related("keystroke_dynamics")
        .order_by("timestamp")
    )
    for s in priors:
        ks = s.keystroke_dynamics.first()
        if not _keystroke_is_plausible(ks):
            continue  # artifact rows are excluded by the live baseline too
        holds.append(ks.avg_hold_time_ms)
        intervals.append(ks.avg_interval_ms)
        cpm.append(ks.typing_speed_cpm)
        if ks.hold_time_std_ms is not None:
            hold_stds.append(ks.hold_time_std_ms)
        if ks.interval_std_ms is not None:
            interval_stds.append(ks.interval_std_ms)
        if ks.longest_pause_ms is not None:
            pauses.append(ks.longest_pause_ms)
        if ks.backspace_count is not None:
            backspaces.append(ks.backspace_count)
    if not holds:
        return None
    return {
        "avg_hold_time_ms": sum(holds) / len(holds),
        "avg_interval_ms": sum(intervals) / len(intervals),
        "typing_speed_cpm": sum(cpm) / len(cpm),
        "hold_time_std_ms":
            sum(hold_stds) / len(hold_stds) if hold_stds else None,
        "interval_std_ms":
            sum(interval_stds) / len(interval_stds) if interval_stds else None,
        "longest_pause_ms":
            sum(pauses) / len(pauses) if pauses else None,
        "backspace_count":
            round(sum(backspaces) / len(backspaces)) if backspaces else None,
    }


def _mid_amount(user) -> str:
    mid = ((user.typical_transfer_min + user.typical_transfer_max) / 2)
    return str(mid.quantize(Decimal("0.01")))


# Reserved marker tower IDs. A session is demo/test debris if and only if its
# ip_or_cell_tower_id is one of these exact strings:
#   * DEMO_TOWER    -- stamped by every Control Room preset payload below;
#   * SMOKE_TOWER   -- stamped by the smoke_test_api.py end-to-end script;
#   * OFFLINE_TOWER -- stamped by offline_fallback.run_demo().
# Tower IDs are stored but never read by the scorer, so the markers are inert
# and safe to use as a purge key. They can never collide with a real row: the
# dataset generators only ever write dotted-quad IPs (app channel) or
# "TWR-" + 8 hex digits (USSD), and every marker contains non-hex text.
DEMO_TOWER = "TWR-DEMO-CONTROL"
SMOKE_TOWER = "TWR-SMOKE-TEST"
OFFLINE_TOWER = "TWR-SMOKE-OFFLINE"
DEBRIS_TOWERS = (DEMO_TOWER, SMOKE_TOWER, OFFLINE_TOWER)


def _purge_previous_demo_traffic():
    """Delete ONLY sessions demonstrably created by demo/test tooling.

    Whitelist-only sweep: a session counts as debris if and only if its
    ip_or_cell_tower_id matches one of DEBRIS_TOWERS. Deleting a session
    cascades to its transactions/features/keystrokes.

    There is deliberately NO same-day-unlabelled heuristic here any more: the
    dataset generator spreads genuine baseline history across a rolling
    ~21-day window, so a meaningful chunk of valid seeded sessions land on
    "today" purely by chance and the old filter wiped them as "debris"
    (measured at 84 rows on a single Control Room load). Genuine rows --
    seeded baselines (IP for app / "TWR-" + 8 hex for USSD), injected attacks
    and anomalies (FraudLabel + real tower/IP), or a live bank customer's
    phone-protected account -- can never equal a marker string, so this sweep
    structurally cannot touch them. Whatever is NOT marker-tagged simply
    survives (fail-safe: at worst we keep an unmarked stray, we never delete
    real data).

    Scoring events are REAL rows once submitted, so re-running the demo
    would otherwise poison velocity/SIM-dominance and shift verdicts on
    stage -- which is exactly why marker-tagged demo rows still MUST be
    cleaned between runs.
    """
    debris = Session.objects.filter(ip_or_cell_tower_id__in=DEBRIS_TOWERS)
    if debris.exists():
        debris.delete()


def get_preset_scenarios() -> dict:
    """Build the five ready-to-submit presets from live database rows."""
    _purge_previous_demo_traffic()

    # Four DISTINCT app users + one USSD user: preset clicks never share
    # behavioural baselines, so rapid-fire demoing stays truthful.
    app_users = _stable_users("app", 4)
    if not app_users:
        raise RuntimeError(
            "No app user with a usable recipient list exists -- presets "
            "need at least one."
        )
    while len(app_users) < 4:
        app_users.append(app_users[-1])
    normal_u, family_u, recovery_u = app_users[0], app_users[2], app_users[3]
    # Patient preset is seeded from the app user with the deepest keystroke
    # history (distinct from the other three) so keystroke_deviation can
    # actually fire in the live demo.  Requires >= MIN_KEYSTROKE_BASELINE_SESSIONS
    # prior keystroke sessions; full confidence around 20.
    patient_u = _best_keystroke_user(
        {normal_u.user_id, family_u.user_id, recovery_u.user_id}
    )
    if patient_u is None:
        patient_u = app_users[1]  # deterministic fallback
    ussd_pool = _stable_users("ussd", 1)
    ussd_user = ussd_pool[0] if ussd_pool else normal_u

    def anchor_for(user, channel):
        a = _baseline_anchor(user, channel)
        if a is None:
            # Pure-USSD users have no fingerprinted rows at all -- fall
            # back to their latest pre-today session of any shape.
            start_today = timezone.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            a = (Session.objects.filter(user=user,
                                        timestamp__lt=start_today)
                 .order_by("-timestamp").first())
        return a

    def in_window_timestamp(user) -> str:
        """Synthesize an event time INSIDE one of the user's usual login
        windows. Needed because presentations happen at arbitrary
        wall-clock hours; the patient/recovery narratives depend on the
        hour being normal so the context-normalcy override can fire.
        The API accepts client timestamps precisely because real
        store-and-forward devices replay past events -- so we replay
        YESTERDAY at the same wall-clock slot: strictly in the past, and
        the hour-of-day lands mid-window."""
        from datetime import timedelta

        now = timezone.now()
        first_window = next(
            ((s, e) for s, e in user.typical_login_hours if s != e),
            (9, 10),  # sensible default slot if windows are degenerate
        )
        start_hour = first_window[0]
        return (now - timedelta(days=1)).replace(
            hour=start_hour, minute=30, second=0, microsecond=0
        ).isoformat()

    def base_payload(user, anchor, channel="app"):
        return {
            "user_id": str(user.user_id),
            "channel": channel,
            "device_fingerprint": (
                anchor.device_fingerprint if channel == "app" else None
            ),
            "sim_id": anchor.sim_id,
            # Marker tower: inert for scoring; lets demo traffic be wiped.
            "ip_or_cell_tower_id": DEMO_TOWER,
            "location_geohash": anchor.location_geohash,
            "session_duration_seconds": 120,
        }

    scenarios = {}

    # ------------------------------------------------------------------
    scenarios["normal_login"] = {
        "scenario_label": "Normal login",
        "scenario_description":
            f"Customer {str(normal_u.user_id)[:8]} logging in from their "
            f"own phone, SIM, location and usual recipient at a typical "
            f"amount. Should sail through.",
        "channel": "app",
        "payload": {
            **base_payload(normal_u, anchor_for(normal_u, "app")),
            "transaction": {
                "amount": _mid_amount(normal_u),
                "recipient_id": normal_u.typical_recipients[0],
            },
        },
    }

    # ------------------------------------------------------------------
    scenarios["obvious_attack"] = {
        "scenario_label": "SIM-swap takeover (USSD)",
        "scenario_description":
            "Attacker on *737# with a cloned SIM: new tower, new location, "
            "5x her largest transfer, brand-new beneficiary. Should be "
            "BLOCKED.",
        "channel": "ussd",
        "payload": {
            **base_payload(ussd_user, anchor_for(ussd_user, "ussd"),
                           channel="ussd"),
            "sim_id": "SIM-attacker-" + ussd_user.user_id.hex[:12],
            "ip_or_cell_tower_id": DEMO_TOWER,
            "location_geohash": "s1tstzz",
            "transaction": {
                "amount": str(ussd_user.typical_transfer_max * 5),
                "recipient_id": "BNF-UNKNOWN-ATTACKER",
            },
        },
    }

    # ------------------------------------------------------------------
    # patient_attack is seeded from the user with the deepest keystroke
    # baseline (see app_users block above); build the attacker's typing
    # rhythm from THAT user's real keystroke profile so the composite
    # deviation registers loudly against it.
    patient_ks = _keystroke_profile(patient_u)
    scenarios["patient_attack"] = {
        "scenario_label": "Patient attacker",
        "scenario_description":
            "Low-and-slow: subtle signals only -- a device the account "
            "never used plus a typing rhythm that does not match the "
            "customer's. SIM, location, hour and amount all stay boringly "
            "normal. The hardest case -- should still draw friction.",
        "channel": "app",
        "payload": {
            **base_payload(patient_u, anchor_for(patient_u, "app")),
            "device_fingerprint":
                "DEV-quiet-" + patient_u.user_id.hex[:16],
            # Replay at her usual banking hour so ONLY the device/rhythm
            # are odd.
            "timestamp": in_window_timestamp(patient_u),
            "transaction": {
                "amount": _mid_amount(patient_u),
                "recipient_id": patient_u.typical_recipients[0],
            },
        },
    }
    if patient_ks is not None:
        # Attacker's rhythm: the takeover runs a scripted/bot keystroke
        # pattern ~50% off this customer's real baseline in every
        # dimension, so keystroke_deviation saturates for the demo.
        _p = patient_ks
        scenarios["patient_attack"]["payload"]["keystroke"] = {
            "avg_hold_time_ms":
                round(max(20.0, _p["avg_hold_time_ms"] * 0.5), 2),
            "hold_time_std_ms":
                round(max(5.0, _p["hold_time_std_ms"] * 0.5), 2)
                if _p["hold_time_std_ms"] is not None else None,
            "avg_interval_ms":
                round(max(30.0, _p["avg_interval_ms"] * 0.5), 2),
            "interval_std_ms":
                round(max(10.0, _p["interval_std_ms"] * 0.5), 2)
                if _p["interval_std_ms"] is not None else None,
            "longest_pause_ms":
                round(max(30.0, _p["longest_pause_ms"] * 0.3), 2)
                if _p["longest_pause_ms"] is not None else None,
            "backspace_count":
                max(1, int(_p["backspace_count"] * 1.5))
                if _p["backspace_count"] is not None else None,
            "typing_speed_cpm":
                round(_p["typing_speed_cpm"] * 2.5, 2),
        }

    # ------------------------------------------------------------------
    scenarios["family_sharing"] = {
        "scenario_label": "Family shared phone",
        "scenario_description":
            "Same phone/SIM/location as always, but a small transfer to a "
            "new person -- mum sending airtime money via daughter's phone. "
            "Legitimate life; at most one gentle verification question.",
        "channel": "app",
        "payload": {
            **base_payload(family_u, anchor_for(family_u, "app")),
            "transaction": {
                "amount": str((family_u.typical_transfer_min / 4)
                              .quantize(Decimal("0.01"))),
                "recipient_id": "BNF-family-friend",
            },
        },
    }

    # ------------------------------------------------------------------
    scenarios["genuine_simswap"] = {
        "scenario_label": "Genuine SIM swap",
        "scenario_description":
            "A REAL customer recovered her stolen phone: new device AND "
            "new SIM, but she's at home, in her hours, paying a known "
            "beneficiary. Friction -- never a frozen account.",
        "channel": "app",
        "payload": {
            **base_payload(recovery_u, anchor_for(recovery_u, "app")),
            "device_fingerprint":
                "DEV-recovery-" + recovery_u.user_id.hex[:12],
            "sim_id": "SIM-new-" + recovery_u.user_id.hex[:16],
            # At home, in her hours: replay inside her usual window so the
            # context-normalcy override can soften this to a challenge.
            "timestamp": in_window_timestamp(recovery_u),
            "transaction": {
                "amount": _mid_amount(recovery_u),
                "recipient_id": recovery_u.typical_recipients[0],
            },
        },
    }

    return scenarios


if __name__ == "__main__":
    import json

    for name, s in get_preset_scenarios().items():
        print(f"\n{name} [{s['channel']}] {s['scenario_label']}")
        print(f"  {s['scenario_description']}")
        print(f"  payload keys: {sorted(s['payload'].keys())}")

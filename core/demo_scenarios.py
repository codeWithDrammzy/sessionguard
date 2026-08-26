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
  patient_attack   -- ONE quiet change only (new device), everything else
                      inside normal range                  -> expect CHALLENGE
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


def _stable_users(channel_preference, need):
    """Up to ``need`` DISTINCT demo-ready users, identity-stable ones
    first (awake preferred inside both groups). Distinctness matters:
    every preset gets its OWN user so clicking several scenarios in a
    row can never let one click's persisted event contaminate another's
    behavioural baseline."""
    hour = timezone.now().hour
    start_today = timezone.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    users = list(BankUser.objects.filter(
        channel_preference=channel_preference).order_by("user_id"))

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


def _mid_amount(user) -> str:
    mid = ((user.typical_transfer_min + user.typical_transfer_max) / 2)
    return str(mid.quantize(Decimal("0.01")))


# Every preset carries this inert marker (tower ids are stored but never
# used by scoring), so repeated demos can be wiped safely.
DEMO_TOWER = "TWR-DEMO-CONTROL"


def _purge_previous_demo_traffic():
    """Delete unlabelled sessions from earlier Control Room runs.

    Two sweeps, both cascading to transactions/features:
      * marker sweep -- anything carrying DEMO_TOWER;
      * orphan sweep -- dev-DB safety net for rows written before the
        marker existed (or by other ad-hoc testing): any SAME-DAY session
        without a FraudLabel. Genuine dataset rows all live inside the
        generated 21-day window and labelled rows are protected, so
        same-day unlabelled traffic can only be test/demo debris.

    Scoring events are REAL rows once submitted, so re-running the demo
    would otherwise poison velocity/SIM-dominance and shift verdicts on
    stage.
    """
    start_today = timezone.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    marked = Session.objects.filter(ip_or_cell_tower_id=DEMO_TOWER)
    orphans = Session.objects.filter(timestamp__gte=start_today).exclude(
        fraud_label__isnull=False
    )
    if marked.exists() or orphans.exists():
        orphans.delete()
        marked.delete()


def get_preset_scenarios() -> dict:
    """Build the five ready-to-submit presets from live database rows."""
    _purge_previous_demo_traffic()

    # Four DISTINCT app users + one USSD user: preset clicks never share
    # behavioural baselines, so rapid-fire demoing stays truthful.
    app_users = _stable_users("app", 4)
    while len(app_users) < 4:
        app_users.append(app_users[-1])
    normal_u, patient_u, family_u, recovery_u = app_users
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
    scenarios["patient_attack"] = {
        "scenario_label": "Patient attacker",
        "scenario_description":
            "Low-and-slow: ONLY the device changed. SIM, location, hour "
            "and amount all stay boringly normal. The hardest case -- "
            "should still draw friction.",
        "channel": "app",
        "payload": {
            **base_payload(patient_u, anchor_for(patient_u, "app")),
            "device_fingerprint":
                "DEV-quiet-" + patient_u.user_id.hex[:16],
            # Replay at her usual banking hour so ONLY the device is odd.
            "timestamp": in_window_timestamp(patient_u),
            "transaction": {
                "amount": _mid_amount(patient_u),
                "recipient_id": patient_u.typical_recipients[0],
            },
        },
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

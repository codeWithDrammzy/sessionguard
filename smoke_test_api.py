"""
Manual end-to-end smoke test for the real-time scoring API.

Runs four scenarios against a live test-client request cycle using REAL
BankUsers already in the dev database:

  1. Approve        -- normal-looking app session for an existing user
                       (reuses their own device/SIM/location/hour window).
  2. Theft mimic    -- new device + new SIM + new location + odd hour +
                       high amount -> expect challenge/block.
  3. Missing fields -- malformed payload -> expect 400.
  4. Unknown user   -- valid UUID not in DB -> expect 404.
  5. Bonus          -- USSD endpoint smoke test (approve-shaped).

Note: scenarios 1/2/5 create real Session rows in the configured database
(they are genuine historical events once scored, by design).

Usage:  python smoke_test_api.py
"""

import json
import os
import sys
import time
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sessionguard_project.settings")
django.setup()

from django.test import Client  # noqa: E402
from django.utils import timezone  # noqa: E402

from core.models import BankUser, Session  # noqa: E402

# Distinctive marker: every session created by this script carries this
# tower id (the field is stored but never used by scoring), so re-runs can
# clean up after themselves and never pollute real history/velocity.
SMOKE_TOWER = "TWR-SMOKE-TEST"


def cleanup_previous_runs():
    """Delete sessions left by earlier smoke runs (cascades to their
    transactions + feature rows) so velocity/history stay pristine.

    Two sweeps:
      * marker sweep -- anything carrying SMOKE_TOWER (this script's own
        rows from previous runs);
      * orphan sweep -- dev-DB safety net for rows written by crashed
        pre-marker runs: sessions stamped today that carry no FraudLabel.
        Genuine dataset rows all live inside the generated 21-day window,
        so a same-day unlabelled row can only be API-test debris.
    """
    start_of_today = timezone.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    stale = Session.objects.filter(ip_or_cell_tower_id=SMOKE_TOWER)
    orphans = Session.objects.filter(
        timestamp__gte=start_of_today
    ).exclude(fraud_label__isnull=False)
    n = stale.count() + orphans.count()
    if n:
        orphans.delete()
        stale.delete()
        print(f"Cleanup: removed {n} session(s) from previous smoke runs.")


def show(title, response, started):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    print(f"HTTP {response.status_code}   ({(time.perf_counter() - started) * 1000:.0f} ms round-trip)")
    body = response.json()
    print(json.dumps(body, indent=2))
    return body


def main():
    cleanup_previous_runs()

    # Warm the model cache NOW so the timed requests below reflect real
    # steady-state latency (a live server pays this once at boot instead).
    from core.ml_model import _load_bundle
    _load_bundle()
    print("Model bundle pre-warmed.")

    client = Client(HTTP_HOST="localhost")  # dev ALLOWED_HOSTS covers this
    hour = timezone.now().hour

    # ---- Scenario 1: pick an app-prefacing user awake at this hour ----
    candidate = None
    for u in BankUser.objects.filter(channel_preference="app"):
        if any(s <= hour < e for s, e in u.typical_login_hours):
            candidate = u
            break
    assert candidate is not None, "No user whose login window covers now."

    # Anchor on a BASELINE session (exclude anything stamped today so a
    # crashed earlier run can never poison the 'normal' reference values).
    start_of_today = timezone.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    anchor = (Session.objects
              .filter(user=candidate, channel="app",
                      timestamp__lt=start_of_today)
              .order_by("-timestamp").first())
    recipient = candidate.typical_recipients[0]
    mid_amount = ((candidate.typical_transfer_min +
                   candidate.typical_transfer_max) / 2).quantize(__import__("decimal").Decimal("0.01"))

    normal_payload = {
        "user_id": str(candidate.user_id),
        "device_fingerprint": anchor.device_fingerprint,
        "sim_id": anchor.sim_id,
        "ip_or_cell_tower_id": SMOKE_TOWER,
        "location_geohash": anchor.location_geohash,
        "session_duration_seconds": 120,
        "transaction": {
            "amount": str(mid_amount),
            "recipient_id": recipient,
        },
    }
    t0 = time.perf_counter()
    r1 = client.post("/api/session-event/", normal_payload,
                     content_type="application/json")
    show(f"TEST 1  APPROVE expected  (user {str(candidate.user_id)[:8]}..., "
         f"known device/SIM/geo, in-window hour {hour} UTC, mid-range amount)",
         r1, t0)

    # ---- Scenario 2: credential-theft mimic on the same user ----
    theft_payload = dict(normal_payload)
    theft_payload.update({
        "device_fingerprint": uuid.uuid4().hex + uuid.uuid4().hex[:32],
        "sim_id": "SIM-" + uuid.uuid4().hex[:32],
        "ip_or_cell_tower_id": SMOKE_TOWER,
        "location_geohash": "s1tstzz",
    })
    theft_payload["transaction"] = {
        "amount": str(candidate.typical_transfer_max * 5),
        "recipient_id": "BNF-UNKNOWN-999",
    }
    t0 = time.perf_counter()
    r2 = client.post("/api/session-event/", theft_payload,
                     content_type="application/json")
    show("TEST 2  BLOCK/CHALLENGE expected  (credential-theft mimic: "
         "new device+SIM+geo+tower, 5x max transfer, new recipient)",
         r2, t0)

    # ---- Scenario 3: missing required fields ----
    t0 = time.perf_counter()
    r3 = client.post("/api/session-event/",
                     {"user_id": str(candidate.user_id)},
                     content_type="application/json")
    show("TEST 3  HTTP 400 expected  (missing sim_id / geo / tower)", r3, t0)

    # ---- Scenario 4: unknown user ----
    ghost = {
        "user_id": str(uuid.uuid4()),
        "sim_id": "SIM-ghost",
        "ip_or_cell_tower_id": SMOKE_TOWER,
        "location_geohash": "s1tst0",
    }
    t0 = time.perf_counter()
    r4 = client.post("/api/session-event/", ghost,
                     content_type="application/json")
    show("TEST 4  HTTP 404 expected  (well-formed UUID, no such BankUser)",
         r4, t0)

    # ---- Bonus: USSD endpoint sanity ----
    ussd_user = None
    for u in BankUser.objects.filter(channel_preference="ussd"):
        if any(s <= hour < e for s, e in u.typical_login_hours):
            ussd_user = u
            break
    if ussd_user is None:
        ussd_user = candidate
    ua = (Session.objects.filter(user=ussd_user, channel="ussd",
                                 timestamp__lt=start_of_today)
          .order_by("-timestamp").first())
    ussd_payload = {
        "user_id": str(ussd_user.user_id),
        "device_fingerprint": "should-be-ignored-on-ussd",
        "sim_id": ua.sim_id,
        "ip_or_cell_tower_id": SMOKE_TOWER,
        "location_geohash": ua.location_geohash,
        "transaction": {"amount": "500.00",
                        "recipient_id": ussd_user.typical_recipients[0]},
    }
    t0 = time.perf_counter()
    r5 = client.post("/api/ussd-event/", ussd_payload,
                     content_type="application/json")
    b5 = show("BONUS  USSD endpoint  (approve expected; fingerprint must "
              "be ignored)", r5, t0)
    assert b5.get("warnings"), "USSD fingerprint warning missing!"

    ok = (r1.status_code == 200 and r1.json()["verdict"] == "approve"
          and r2.status_code == 200 and r2.json()["verdict"] != "approve"
          and r3.status_code == 400 and r4.status_code == 404
          and r5.status_code == 200 and r5.json()["verdict"] == "approve")
    print(f"\n{'=' * 74}\nSMOKE TEST {'PASSED' if ok else 'FAILED'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

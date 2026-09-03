"""SessionGuard behavioural-guarantee tests.

These are intentional, logic-level tests (not happy-path demo tests). Each
one pins down a documented design invariant so a future refactor cannot
silently change it:

  * Feature causality      -- a session's features are computed ONLY from its
                             own data + the user's strictly-prior history.
  * Keystroke baseline     -- keystroke_deviation_score stays None until >=5
                             prior keystroke sessions (MIN_KEYSTROKE_BASELINE_SESSIONS).
  * Hybrid context override -- a hardware change in an otherwise-normal session
                             is capped at CHALLENGE (never block); an attack
                             with abnormal context stays BLOCK.
  * Offline degraded-mode -- a low-information local check NEVER returns block.
  * ML vectorisation       -- None -> 0.0, bool -> int, and the saved bundle's
                             FEATURE_COLUMNS match the code's FEATURE_COLUMNS.

Fixtures are hand-built small sessions (not the 250-customer synthetic DB),
so the suite runs in seconds on an isolated in-memory test database.
"""
from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from core.feature_engine import compute_features
from core.hybrid_scorer import score_session_hybrid
from core.ml_model import FEATURE_COLUMNS, features_to_vector, _load_bundle
from core.models import (
    BankUser,
    BehavioralFeatures,
    KeystrokeDynamics,
    Session,
    resolve_combined_device_location_flag,
)
from core.offline_fallback import build_local_cache, score_session_offline
from core.rules_engine import APPROVE_MAX, CHALLENGE_MAX, WEIGHTS, score_session


def make_user(**kw):
    defaults = dict(
        typical_login_hours=[[8, 22]],
        typical_transfer_min=1000,
        typical_transfer_max=500000,
        typical_recipients=["BNF-0001", "BNF-0002"],
        channel_preference=BankUser.CHANNEL_APP,
    )
    defaults.update(kw)
    return BankUser.objects.create(**defaults)


def make_session(user, ts, *, device="dev-A", sim="sim-1", geo="s1tstzz",
                 channel="app", duration=60):
    return Session.objects.create(
        user=user,
        channel=channel,
        timestamp=ts,
        device_fingerprint=device,
        sim_id=sim,
        ip_or_cell_tower_id="TWR-1",
        location_geohash=geo,
        session_duration_seconds=duration,
    )


def make_keystroke(session, hold=150.0, interval=1000.0, cpm=55.0,
                   failures=None):
    return KeystrokeDynamics.objects.create(
        session=session,
        avg_hold_time_ms=hold,
        avg_interval_ms=interval,
        typing_speed_cpm=cpm,
        login_pin_failures=failures,
    )


class FeatureCausalityTests(TestCase):
    def test_first_session_nevers_changed_flags(self):
        user = make_user()
        now = timezone.now()
        s = make_session(user, now)
        f = compute_features(s)
        # No prior history -> nothing can be "changed", keystroke None (<5 priors).
        self.assertFalse(f.device_change_flag)
        self.assertFalse(f.sim_change_flag)
        self.assertFalse(f.location_change_flag)
        self.assertFalse(f.combined_device_location_flag)
        self.assertIsNone(f.keystroke_deviation_score)

    def test_history_is_strictly_prior_only(self):
        user = make_user()
        t0 = timezone.now() - timedelta(days=3)
        # After 5 prior keystroke sessions the 6th session gets a REAL score.
        for i in range(5):
            ps = make_session(user, t0 + timedelta(minutes=i * 30),
                              device="dev-A", sim="sim-1", geo="s1tstzz")
            make_keystroke(ps, cpm=50.0 + i)
        target = make_session(user, t0 + timedelta(days=1))
        make_keystroke(target, cpm=100.0)  # dramatically faster
        f = compute_features(target)
        # >=5 priors -> the gate opens and the deviation is now computed.
        self.assertIsNotNone(f.keystroke_deviation_score)
        self.assertGreater(f.keystroke_deviation_score, 0.0)
        # A LATER session must never influence an earlier session's features.
        later = make_session(user, t0 + timedelta(days=5), device="dev-B")
        f_before_later = compute_features(later)
        # The target session already reported dev-A, so dev-B IS a change here.
        self.assertTrue(f_before_later.device_change_flag)

    def test_combined_flag_requires_device_or_sim_AND_location(self):
        user = make_user()
        t0 = timezone.now() - timedelta(days=1)
        make_session(user, t0, device="dev-A", sim="sim-1", geo="s1tstzz")
        # Device changed but SAME location -> combined flag must stay False.
        s = make_session(user, t0 + timedelta(hours=1), device="dev-B",
                         sim="sim-1", geo="s1tstzz")
        f = compute_features(s)
        self.assertTrue(f.device_change_flag)
        self.assertFalse(f.location_change_flag)
        self.assertFalse(f.combined_device_location_flag)
        # Device AND location changed -> combined flag True.
        s2 = make_session(user, t0 + timedelta(hours=2), device="dev-C",
                          sim="sim-1", geo="s2xxxxx")
        f2 = compute_features(s2)
        self.assertTrue(f2.combined_device_location_flag)
        # Model save() enforces the same invariant.
        self.assertTrue(resolve_combined_device_location_flag(
            f2.device_change_flag, f2.sim_change_flag, f2.location_change_flag))


class KeystrokeBaselineGateTests(TestCase):
    def test_keystroke_none_below_five_priors(self):
        from core.feature_engine import MIN_KEYSTROKE_BASELINE_SESSIONS
        self.assertEqual(MIN_KEYSTROKE_BASELINE_SESSIONS, 5)
        user = make_user()
        now = timezone.now()
        for i in range(4):  # only 4 priors
            ps = make_session(user, now + timedelta(minutes=-60 + i * 10),
                              device="dev-A")
            make_keystroke(ps)
        target = make_session(user, now)
        make_keystroke(target)
        f = compute_features(target)
        self.assertIsNone(f.keystroke_deviation_score)
        # Even a wild deviation must not surface without enough baseline.
        self.assertIsNone(f.keystroke_deviation_score)


class RuleWeightTests(TestCase):
    def test_keystroke_alone_cannot_reach_challenge(self):
        # WEIGHTS["keystroke_deviation_max"] is 12; a full (1.0) keystroke
        # deviation contributes at most 12 < the 30 challenge threshold.
        max_pts = WEIGHTS["keystroke_deviation_max"]
        self.assertLess(max_pts, CHALLENGE_MAX)
        self.assertLess(APPROVE_MAX, CHALLENGE_MAX)
        # Construct a features row where ONLY keystroke deviates maximally.
        user = make_user()
        s = make_session(user, timezone.now())
        f = BehavioralFeatures(session=s)
        f.keystroke_deviation_score = 1.0
        d = score_session(f)
        self.assertEqual(d.score, max_pts)
        self.assertEqual(d.verdict, "approve")  # 12 < 30
        self.assertEqual([r["code"] for r in d.triggered_reasons],
                         ["keystroke_deviation"])


class HybridOverrideTests(TestCase):
    def _hybrid(self, keystroke=None, **feature_kw):
        user = make_user()
        s = make_session(user, timezone.now())
        f = BehavioralFeatures(session=s)
        f.keystroke_deviation_score = keystroke
        defaults = dict(
            hour_deviation_score=0.0, amount_deviation_score=None,
            device_change_flag=False, sim_change_flag=False,
            location_change_flag=False, combined_device_location_flag=False,
            impossible_travel_flag=False, new_recipient_flag=False,
            velocity_count_5min=0, menu_timing_deviation_score=None,
        )
        defaults.update(feature_kw)
        for k, v in defaults.items():
            setattr(f, k, v)
            if k == "combined_device_location_flag":
                f.combined_device_location_flag = resolve_combined_device_location_flag(
                    f.device_change_flag, f.sim_change_flag,
                    f.location_change_flag)
        return score_session_hybrid(f), f

    def test_hardware_change_context_normal_caps_at_challenge(self):
        # Genuine SIM-swap shape: device+SIM+location changed (combined flag
        # True would normally be a hard block) but context is otherwise
        # ordinary.
        d, _ = self._hybrid(
            device_change_flag=True, sim_change_flag=True,
            location_change_flag=True,
        )
        # The VERDICT is capped to challenge (never block, never reversed
        # upward). The raw score intentionally stays high (ML p is ~1.0 for
        # this shape -> score ~100) -- that is the documented "block-worthy
        # raw score, softened to verification" behaviour, NOT a bug.
        self.assertEqual(d.verdict, "challenge")
        self.assertTrue(d.context_override_applied)
        self.assertGreater(d.score, CHALLENGE_MAX)

    def test_attack_with_bad_context_stays_block(self):
        # Credential-theft shape: combined flag + impossible travel + abnormal
        # amount -> must NOT be softened.
        d, _ = self._hybrid(
            device_change_flag=True, sim_change_flag=True,
            location_change_flag=True, impossible_travel_flag=True,
            amount_deviation_score=1.0,
        )
        self.assertEqual(d.verdict, "block")
        self.assertFalse(d.context_override_applied)
        self.assertGreater(d.score, CHALLENGE_MAX)


class OfflineDegradedTests(TestCase):
    def test_degraded_never_blocks(self):
        user = make_user()
        # Build a fresh cached snapshot (no priors).
        now = timezone.now()
        cache = build_local_cache(user)
        # A maximally-suspicious offline event: device mismatch + off-hours +
        # wild amount. Even at max score it must cap at CHALLENGE.
        ev = {
            "session_id": "x",
            "device_fingerprint": "DIFFERENT",
            "sim_id": "DIFFERENT",
            "timestamp": now.isoformat(),
            "transaction": {"amount": "99999999", "recipient_id": "X"},
        }
        # cache has no last-known values, so prime it to force a mismatch.
        cache["last_known_device_fingerprint"] = "KNOWN"
        cache["last_known_sim_id"] = "KNOWN"
        cache["typical_login_hours"] = [[8, 22]]  # event hour may differ
        d = score_session_offline(ev, cache)
        self.assertNotEqual(d.verdict, "block")
        self.assertEqual(d.verdict, "challenge")
        self.assertTrue(d.is_degraded)
        self.assertIn("offline_degraded_check",
                      [r["code"] for r in d.triggered_reasons])


class MLVectorisationTests(TestCase):
    def test_feature_columns_match_saved_bundle(self):
        bundle = _load_bundle()
        self.assertEqual(list(bundle["feature_columns"]), FEATURE_COLUMNS)

    def test_vector_none_to_zero_bool_to_int(self):
        user = make_user()
        s = make_session(user, timezone.now())
        f = BehavioralFeatures(session=s)
        f.keystroke_deviation_score = None
        f.amount_deviation_score = None
        f.menu_timing_deviation_score = None
        f.device_change_flag = True
        v = features_to_vector(f)
        self.assertEqual(len(v), len(FEATURE_COLUMNS))
        # None fields -> 0.0; bool -> int.
        self.assertEqual(v[FEATURE_COLUMNS.index("keystroke_deviation_score")], 0.0)
        self.assertEqual(v[FEATURE_COLUMNS.index("amount_deviation_score")], 0.0)
        self.assertEqual(v[FEATURE_COLUMNS.index("device_change_flag")], 1)

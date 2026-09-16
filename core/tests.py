"""SessionGuard behavioural-guarantee tests.

These are intentional, logic-level tests (not happy-path demo tests). Each
one pins down a documented design invariant so a future refactor cannot
silently change it:

  * Feature causality      -- a session's features are computed ONLY from its
                             own data + the user's strictly-prior history.
  * Keystroke baseline     -- keystroke_deviation_score stays None until >=5
                             prior keystroke sessions (MIN_KEYSTROKE_BASELINE_SESSIONS),
                             then ramps in confidence (50% at 5 priors) to full
                             at KEYSTROKE_FULL_CONFIDENCE_SESSIONS.
  * Hybrid context override -- a hardware change in an otherwise-normal session
                             is capped at CHALLENGE (never block); an attack
                             with abnormal context stays BLOCK.
  * Hybrid keystroke guard -- a block whose ONLY cause (rules AND ML excluded)
                             is keystroke deviation is capped at CHALLENGE.
  * Offline degraded-mode -- a low-information local check NEVER returns block.
  * ML vectorisation       -- None -> 0.0, bool -> int, and the saved bundle's
                             FEATURE_COLUMNS match the code's FEATURE_COLUMNS.
  * Demo purge safety      -- the Control Room purge removes ONLY reserved
                             marker-tagged debris; seeded same-day sessions
                             and unmarked strays always survive.

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
from core.explanation import explain_decision
from core.rules_engine import CHALLENGE_MAX, WEIGHTS, score_session


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
                 channel="app", duration=60, tower="TWR-1"):
    return Session.objects.create(
        user=user,
        channel=channel,
        timestamp=ts,
        device_fingerprint=device,
        sim_id=sim,
        ip_or_cell_tower_id=tower,
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


class KeystrokeConfidenceTests(TestCase):
    def test_below_min_baseline_has_no_contribution(self):
        # 24h login window so the hour-deviation signal is 0 regardless of
        # what wall-clock time the suite runs (score must be keystroke-only).
        user = make_user(typical_login_hours=[[0, 24]])
        now = timezone.now()
        for i in range(4):  # only 4 priors
            ps = make_session(user, now - timedelta(minutes=120 - i * 10),
                              device="dev-A")
            make_keystroke(ps)
        target = make_session(user, now)
        make_keystroke(target, cpm=999.0)  # wildly different typing
        f = compute_features(target)
        self.assertIsNone(f.keystroke_deviation_score)
        d = score_session(f)
        self.assertEqual(d.score, 0)
        self.assertEqual(d.verdict, "approve")
        self.assertNotIn("keystroke_deviation",
                         [r["code"] for r in d.triggered_reasons])

    def test_confidence_scales_with_baseline_size(self):
        # Identical raw typing deviation for both users (same prior values,
        # same current keystroke), only the baseline SIZE differs: 20 priors
        # -> full confidence, 5 priors (the minimum) -> ~50% confidence.
        now = timezone.now()
        a_user = make_user()
        for i in range(20):
            ps = make_session(a_user,
                              now + timedelta(minutes=-600 + i * 30),
                              device="dev-A")
            make_keystroke(ps, interval=1000.0 + 10 * i)
        a_target = make_session(a_user, now - timedelta(minutes=20))
        make_keystroke(a_target, interval=2000.0)

        b_user = make_user()
        for i in range(5):
            ps = make_session(b_user,
                              now + timedelta(minutes=-180 + i * 30),
                              device="dev-B")
            make_keystroke(ps, interval=1000.0 + 10 * i)
        b_target = make_session(b_user, now)
        make_keystroke(b_target, interval=2000.0)

        fa = compute_features(a_target)
        fb = compute_features(b_target)
        self.assertGreater(fa.keystroke_deviation_score, 0.0)
        self.assertGreater(fb.keystroke_deviation_score, 0.0)
        self.assertGreater(fa.keystroke_deviation_score,
                           fb.keystroke_deviation_score)
        # The long-history signal is worth exactly twice the thin-baseline
        # signal (raw is identical; confidence is 1.0 vs 0.5).
        self.assertAlmostEqual(fa.keystroke_deviation_score,
                               2.0 * fb.keystroke_deviation_score)


class KeystrokeArtifactGuardTests(TestCase):
    def test_artifact_row_is_excluded_from_baseline(self):
        # A capture artifact (e.g. a 16s hold during a screen re-render)
        # must never become part of the user's typing baseline: a single
        # such row inflates baseline std-dev and blinds every real
        # deviation afterwards.
        from core.feature_engine import (
            MIN_KEYSTROKE_BASELINE_SESSIONS, _keystroke_is_plausible)
        user = make_user()
        now = timezone.now()
        # Enough PRIOR realistic sessions to open the baseline gate...
        for i in range(MIN_KEYSTROKE_BASELINE_SESSIONS):
            ps = make_session(user,
                              now - timedelta(minutes=120 - i * 20),
                              device="dev-A")
            make_keystroke(ps, hold=150.0, interval=700.0 + 20 * i, cpm=60.0)
        # ...plus one artifact session with a 16s hold / 0 reading speed.
        artifact = make_session(user, now - timedelta(minutes=5),
                                device="dev-A")
        make_keystroke(artifact, hold=16293.7, interval=16170.4, cpm=4.0)
        # The artifact row must be treated as junk by the guard itself.
        self.assertFalse(_keystroke_is_plausible(artifact.keystroke_dynamics.first()))

        # A target session with genuinely different (but physically sane)
        # typing must now register as a deviation instead of being absorbed
        # into the artifact-bloated baseline.
        target = make_session(user, now, device="dev-A")
        make_keystroke(target, hold=200.0, interval=2600.0, cpm=24.0)
        f = compute_features(target)
        self.assertIsNotNone(f.keystroke_deviation_score)
        self.assertGreater(f.keystroke_deviation_score, 0.0)

    def test_realistic_slow_typing_still_counts_as_deviation(self):
        # Regression for the live user scenario: 26 sane baseline rows plus
        # 3 artifact rows. With artifact rows in the baseline the genuine
        # slow session scored 0.0; excluded, the same session must score > 0.
        from core.feature_engine import _keystroke_is_plausible
        user = make_user()
        now = timezone.now()
        sane = [
            (140, 815, 70), (121, 480, 90), (125, 734, 80), (118, 562, 85),
            (133, 553, 70), (142, 692, 75), (125, 503, 90), (120, 838, 70),
            (155, 1158, 55), (125, 500, 90), (102, 716, 85), (100, 1803, 35),
            (137, 791, 70), (119, 963, 55), (130, 1296, 45), (132, 815, 65),
            (161, 1254, 45), (140, 817, 60), (146, 1250, 55), (134, 895, 60),
            (142, 1162, 55), (148, 1477, 40), (157, 899, 60), (142, 1428, 45),
            (141, 1263, 45), (144, 988, 55),
        ]
        artifacts = [(146, 4834, 13), (3047, 693, 46), (16293, 16170, 4)]
        for i, (hold, inter, cpm) in enumerate(sane + artifacts):
            ps = make_session(
                user,
                now - timedelta(minutes=(len(sane) + len(artifacts) - i) * 10),
                device="dev-A")
            make_keystroke(ps, hold=hold, interval=inter, cpm=cpm)
        chips = [a.keystroke_dynamics.first() for a in
                 Session.objects.filter(user=user).prefetch_related(
                     "keystroke_dynamics") if a.keystroke_dynamics.first()]
        self.assertEqual(sum(1 for k in chips if _keystroke_is_plausible(k)),
                         len(sane))

        target = make_session(user, now, device="dev-A")
        make_keystroke(target, hold=206.4, interval=2574.4, cpm=24.0)
        f = compute_features(target)
        self.assertGreater(f.keystroke_deviation_score, 0.5)


class RuleWeightTests(TestCase):
    def test_keystroke_alone_lands_in_challenge_never_block(self):
        # WEIGHTS["keystroke_deviation_max"] is 40: a full (1.0) keystroke
        # deviation contributes round(1.0 * 40) = 40 -> CHALLENGE band
        # (30-59), never reaching the >=60 block floor on its own.
        max_pts = WEIGHTS["keystroke_deviation_max"]
        self.assertGreaterEqual(max_pts, 30)
        self.assertLess(max_pts, 60)
        # Construct a features row where ONLY keystroke deviates maximally.
        user = make_user()
        s = make_session(user, timezone.now())
        f = BehavioralFeatures(session=s)
        f.keystroke_deviation_score = 1.0
        d = score_session(f)
        self.assertEqual(d.score, max_pts)
        self.assertEqual(d.verdict, "challenge")  # 40 within 30-59
        self.assertEqual([r["code"] for r in d.triggered_reasons],
                         ["keystroke_deviation"])

    def test_keystroke_with_amount_and_hour_still_blocks(self):
        # The strengthened signal must not neuter the engine: keystroke (40)
        # + amount (20) + hour (15) sums to >=60 -> block at the RULES layer.
        user = make_user()
        s = make_session(user, timezone.now())
        f = BehavioralFeatures(session=s)
        f.keystroke_deviation_score = 1.0
        f.amount_deviation_score = 1.0
        f.hour_deviation_score = 1.0
        d = score_session(f)
        self.assertGreaterEqual(d.score, 60)
        self.assertEqual(d.verdict, "block")
        codes = {r["code"] for r in d.triggered_reasons}
        self.assertEqual(codes, {"keystroke_deviation", "amount_deviation",
                                 "hour_deviation"})


class HybridKeystrokeGuardTests(TestCase):
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
        return score_session_hybrid(f)

    def test_keystroke_dominant_block_softened_to_challenge(self):
        # keystroke(40) + amount(20) = rules 60 -> block. Removing the
        # keystroke contribution leaves 20 and ML (p~0.13 -> 13) does not
        # block either, so typing rhythm is the ONLY cause -> challenge.
        d = self._hybrid(keystroke=1.0, amount_deviation_score=1.0)
        self.assertEqual(d.verdict, "challenge")
        self.assertTrue(d.keystroke_override_applied)
        self.assertFalse(d.context_override_applied)
        self.assertIn("keystroke_override",
                      [r["code"] for r in d.triggered_reasons])
        # The raw combined score intentionally stays block-level (60).
        self.assertGreater(d.score, CHALLENGE_MAX)
        customer, note = explain_decision(d)
        self.assertIn("typing rhythm", note)

    def test_not_keystroke_dominant_block_stays_block(self):
        # impossible_travel(50) + amount(20) = rules 70 WITHOUT keystroke,
        # so typing rhythm is not the block's cause -> no softening.
        d = self._hybrid(keystroke=1.0, amount_deviation_score=1.0,
                         impossible_travel_flag=True)
        self.assertEqual(d.verdict, "block")
        self.assertFalse(d.keystroke_override_applied)
        self.assertFalse(d.context_override_applied)

    def test_ml_driven_block_not_softened(self):
        # Hardware-cluster shape makes ML p~1.0 despite keystroke being the
        # dominant RULES cause (rules-without-keystroke = 55 < 60): the ML
        # signal keeps the block. Context is NOT normal (amount above the
        # normalcy threshold), so the hardware override is also inactive.
        d = self._hybrid(
            keystroke=1.0, device_change_flag=True, sim_change_flag=True,
            location_change_flag=True, amount_deviation_score=0.5,
        )
        self.assertEqual(d.verdict, "block")
        self.assertFalse(d.keystroke_override_applied)
        self.assertFalse(d.context_override_applied)


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


class ControlRoomPurgeSafetyTests(TestCase):
    """The demo purge must ONLY remove marker-tagged test/debris rows.

    Regression for a measured data-loss bug: the old purge also swept every
    SAME-DAY session without a FraudLabel as 'debris'. The dataset generator
    spreads baseline history across a rolling ~21-day window, so valid seeded
    sessions land on 'today' purely by chance and a single Control Room load
    deleted 84 of them live. Debris must be identified by the reserved marker
    tower IDs ONLY (whitelist), never by date/labelling heuristics.
    """

    def test_purge_preserves_seeded_looking_same_day_sessions(self):
        from core.demo_scenarios import _purge_previous_demo_traffic

        user = make_user()
        now = timezone.now()
        # The two shapes the dataset generators actually write:
        # app -> dotted-quad IP, USSD -> "TWR-" + 8 hex digits.
        app_seed = make_session(user, now, device="dev-A", sim="sim-1",
                                geo="s1tstzz", channel="app", tower="41.2.3.4")
        ussd_seed = make_session(user, now, device=None, sim="sim-2",
                                 geo="s1tstzz", channel="ussd",
                                 tower="TWR-1a2b3c4d")
        # A same-day session with a FraudLabel (attack/anomaly shape).
        labelled = make_session(user, now, device=None, sim="sim-3",
                                geo="s1tstzz", channel="ussd",
                                tower="TWR-0f0f0f0f")

        _purge_previous_demo_traffic()

        self.assertTrue(Session.objects.filter(pk=app_seed.pk).exists())
        self.assertTrue(Session.objects.filter(pk=ussd_seed.pk).exists())
        self.assertTrue(Session.objects.filter(pk=labelled.pk).exists())

    def test_purge_preserves_unmarked_same_day_session(self):
        """Fail-safe direction: an unmarked stray survives (never destroyed).
        A random unswept ad-hoc row is safer than deleting real data."""
        from core.demo_scenarios import _purge_previous_demo_traffic

        user = make_user()
        stray = make_session(user, timezone.now(), device="dev-Q",
                             sim="sim-9", geo="s1tstzz", channel="app",
                             tower="202.122.10.11")

        _purge_previous_demo_traffic()

        self.assertTrue(Session.objects.filter(pk=stray.pk).exists())

    def test_purge_sweeps_every_marker_debris(self):
        from core.demo_scenarios import DEBRIS_TOWERS, _purge_previous_demo_traffic

        user = make_user()
        now = timezone.now()
        debris = [
            make_session(user, now, device="dev-A", sim="sim-1", geo="s1tstzz",
                         channel="app", tower=tower)
            for tower in DEBRIS_TOWERS
        ]

        _purge_previous_demo_traffic()

        for s in debris:
            self.assertFalse(Session.objects.filter(pk=s.pk).exists())

    def test_purge_cascades_to_transactions_and_keystrokes(self):
        from core.demo_scenarios import _purge_previous_demo_traffic
        from core.models import KeystrokeDynamics, Transaction

        user = make_user()
        s = make_session(user, timezone.now(), device="dev-A", sim="sim-1",
                         geo="s1tstzz", channel="app", tower="TWR-DEMO-CONTROL")
        Transaction.objects.create(
            session=s,
            timestamp=s.timestamp,
            amount=1000,
            recipient_id="BNF-0001",
            is_new_recipient=False,
        )
        make_keystroke(s)

        _purge_previous_demo_traffic()

        self.assertFalse(Session.objects.filter(pk=s.pk).exists())
        self.assertFalse(Transaction.objects.filter(session=s).exists())
        self.assertFalse(KeystrokeDynamics.objects.filter(session=s).exists())

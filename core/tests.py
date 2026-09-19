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
from core.hybrid_scorer import HybridDecision, score_session_hybrid
from core.ml_model import FEATURE_COLUMNS, features_to_vector, _load_bundle
from core.models import (
    BankUser,
    BehavioralFeatures,
    KeystrokeDynamics,
    Session,
    Transaction,
    resolve_combined_device_location_flag,
)
from core.offline_fallback import build_local_cache, score_session_offline
from core.explanation import explain_decision
from core.rules_engine import CHALLENGE_MAX, WEIGHTS, score_session
from core.secondary_behavior import (
    CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE,
    ESCALATE_IF_FAMILIARITY_BELOW,
)


def make_user(**kw):
    """Create a BankUser with sane default habits, overridable per test."""
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
    """Create a Session row with stable per-test defaults."""
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
                   failures=None, hold_std=None, interval_std=None,
                   longest_pause=None, backspaces=None):
    """Create a KeystrokeDynamics row for a session with stable defaults."""
    return KeystrokeDynamics.objects.create(
        session=session,
        avg_hold_time_ms=hold,
        hold_time_std_ms=hold_std,
        avg_interval_ms=interval,
        interval_std_ms=interval_std,
        longest_pause_ms=longest_pause,
        backspace_count=backspaces,
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


class KeystrokeSensitivityTests(TestCase):
    """Richer-feature composite: strong single-dimension and rhythm-shape
    deviations must fully surface instead of being diluted by a plain mean."""

    def _baseline_and_target(self, priors, target_ks):
        user = make_user()
        now = timezone.now()
        for i, ks in enumerate(priors):
            ps = make_session(user, now - timedelta(minutes=600 - i * 30),
                              device="dev-A")
            make_keystroke(ps, **ks)
        t = make_session(user, now, device="dev-A")
        make_keystroke(t, **target_ks)
        return compute_features(t).keystroke_deviation_score

    def test_fast_typist_no_longer_diluted_to_zero(self):
        # Live regression: cpm=114 vs owner baseline ~70 (z=+1.56) with
        # hold/int near-normal scored 0.0127 under the old mean-of-|z|.
        # The max-blend composite must keep a single strong dimension loud.
        avg = [dict(hold=137.0, interval=1054.0, cpm=70.0)
               for _ in range(20)]
        score = self._baseline_and_target(
            avg, dict(hold=126.8, interval=542.8, cpm=114.0))
        self.assertGreater(score, 0.1)
        self.assertLessEqual(score, 1.0)

    def test_rhythm_shape_dimensions_raise_the_signal(self):
        # Same unrevealing MEANS but a completely different rhythm SHAPE
        # (highly unstable holds/intervals, long hunting pauses, edits).
        # Without the richer dimensions this reads as ~0; with them it
        # must register.
        avg = [dict(hold=140.0, interval=900.0, cpm=60.0,
                    hold_std=18.0, interval_std=200.0,
                    longest_pause=2600.0, backspaces=1)
               for _ in range(20)]
        score = self._baseline_and_target(
            avg, dict(hold=140.0, interval=900.0, cpm=60.0,
                      hold_std=95.0, interval_std=1100.0,
                      longest_pause=8600.0, backspaces=9))
        self.assertGreater(score, 0.2)
        self.assertLessEqual(score, 1.0)

    def test_login_pin_failures_floor_escalates(self):
        # Three wrong-PIN attempts before success is nearly never "just a
        # flub" -- the rightful owner knows the PIN. Even with typing that
        # otherwise matches baseline, 3+ failures must push the score up.
        varied = [dict(hold=137.0 + i, interval=1054.0 + 40 * i, cpm=70.0 + i)
                  for i in range(20)]
        score = self._baseline_and_target(
            varied, dict(hold=135.0, interval=1000.0, cpm=72.0, failures=3))
        self.assertGreater(score, 0.5)
        # A single flub stays within human-error territory: no floor.
        score1 = self._baseline_and_target(
            varied[:], dict(hold=135.0, interval=1000.0, cpm=72.0,
                            failures=1))
        self.assertLess(score1, 0.5)


class KeystrokePoisonGuardTests(TestCase):
    """Regression for the repeated-attacker cascade: a session the system
    flagged as a keystroke anomaly must NOT enter the typing baseline,
    otherwise the NEXT identical attacker scores below threshold (matching
    the polluted "normal") and is silently approved. Live path uses the
    per-session stored score; batch uses the just-computed score.

    Tuned parameters: owner baseline spread (25ms hold, 300ms interval,
    12 cpm) and attacker (154ms hold, 1250ms interval, 50 cpm) give
    first-attack f1=0.525 (caught, > 0.5 threshold) but second-attack
    f2=0.428 (missed) IF a1 was absorbed -- and 0.525 WITH the guard.
    That is precisely the user-observed: works once, then goes silent."""

    ABS = None  # set from feature_engine below

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from core.feature_engine import KEYSTROKE_BASELINE_ABSORB_THRESHOLD
        cls.ABS = KEYSTROKE_BASELINE_ABSORB_THRESHOLD

    def _seed_user(self, n=26):
        """Owner baseline of normal typing; returns (user, priors, now)."""
        user = make_user()
        now = timezone.now()
        import random
        rng = random.Random(7)
        priors = []
        for i in range(n):
            ps = make_session(
                user,
                now - timedelta(minutes=(n - i + 2) * 10),  # strict prior
                device="dev-A")
            make_keystroke(ps,
                           hold=140.0 + rng.uniform(-25, 25),
                           interval=900.0 + rng.uniform(-300, 300),
                           cpm=60.0 + rng.uniform(-12, 12))
            priors.append(ps)
        return user, priors, now

    def _attack(self, user, ts):
        s = make_session(user, ts, device="dev-A")
        make_keystroke(s, hold=154.0, interval=1250.0, cpm=50.0)
        return s

    def _score_before(self, user_id, session):
        from core.feature_engine import compute_features, load_user_history
        h = load_user_history(user_id, before=session.timestamp)
        return compute_features(session, history=h).keystroke_deviation_score

    def test_guard_keeps_baseline_clean_across_attackers(self):
        # Live path: a1's scored features are PERSISTED (a BehavioralFeatures
        # row, as score_session does), so load_user_history reads back its
        # offending score and excludes a1 from the baseline. The second,
        # identical attacker must still be flagged.
        user, priors, now = self._seed_user()

        a1 = self._attack(user, now)
        f1 = self._score_before(user.user_id, a1)
        self.assertGreater(f1, self.ABS)          # caught on arrival
        f1_row = BehavioralFeatures(session=a1,
                                    keystroke_deviation_score=f1)
        f1_row.save()

        a2 = self._attack(user, now + timedelta(minutes=5))
        f2 = self._score_before(user.user_id, a2)
        self.assertGreater(f2, self.ABS)          # still caught

    def test_without_guard_pollution_suppresses_second_attack(self):
        # Pre-guard / if the score row is missing: load_user_history treats a1
        # as teachable (keystroke_deviation=None -> _keystroke_absorb_ok(True))
        # so a1's novel rhythm folds into "normal" and the identical second
        # attacker slips under the threshold. This is the exact regression.
        user, priors, now = self._seed_user()

        a1 = self._attack(user, now)
        a2 = self._attack(user, now + timedelta(minutes=5))

        f2 = self._score_before(user.user_id, a2)  # a1 unrecorded -> polluted
        # Compare to the guarded outcome: a1 excluded -> second still caught.
        f2_clean = self._score_with_guard(user.user_id, a2, excluded={a1.pk})
        self.assertLess(f2, self.ABS)             # polluted: silent approve
        self.assertGreater(f2_clean, self.ABS)    # guarded: still caught

    def _score_before(self, user_id, session):
        from core.feature_engine import compute_features, load_user_history
        h = load_user_history(user_id, before=session.timestamp)
        return compute_features(session, history=h).keystroke_deviation_score

    def _score_with_guard(self, user_id, session, excluded):
        """Like live scoring but forced-guarded: replay history from DB,
        dropping the excluded sessions entirely (they would have been
        kept-out by the absorb guard)."""
        from core.feature_engine import (
            compute_features, UserHistory, _keystroke_is_plausible)
        from core.models import Session
        h = UserHistory()
        priors = Session.objects.filter(
            user_id=user_id, timestamp__lt=session.timestamp,
        ).prefetch_related("keystroke_dynamics").order_by("timestamp")
        for s in priors:
            if s.pk in excluded:
                continue
            ks = s.keystroke_dynamics.first()
            if ks is not None and _keystroke_is_plausible(ks):
                h.observe(s)
        return compute_features(session, history=h).keystroke_deviation_score


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


class BalanceCheckNeverScoredTests(TestCase):
    """A read-only balance check must NEVER be run through the fraud-scoring
    pipeline. Regression for the USSD dead-end: after a cloned-SIM/SIM-switch
    scenario the old flow scored 'Check Balance' (no transaction) and a
    challenge verdict left the customer with no OTP path -- just a 'Balance
    unavailable' message. Reading your own balance moves no money, so it is
    always permitted regardless of device/SIM/location change."""

    def test_balance_check_with_fresh_sim_always_approves(self):
        user = make_user()
        # Every signal screams 'unseen device+SIM+location' -- exactly the
        # shape that WOULD challenge/block a scored transfer.
        res = self.client.post(
            "/api/bank/send-money/",
            {
                "user_id": str(user.user_id),
                "channel": "app",
                "device_fingerprint": "dev-fresh",
                "sim_id": "SIM-fresh-swap",
                "ip_or_cell_tower_id": "41.2.3.4",
                "location_geohash": "s1tstzz",
            },
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["verdict"], "approve")
        self.assertIn("balance", body)

    def test_balance_check_creates_no_session(self):
        """Proof it bypassed scoring entirely: no Session is persisted."""
        user = make_user()
        self.client.post(
            "/api/bank/send-money/",
            {
                "user_id": str(user.user_id),
                "channel": "ussd",
                "device_fingerprint": None,
                "sim_id": "SIM-fresh-swap",
                "ip_or_cell_tower_id": "TWR-deadbeef",
                "location_geohash": "s1tstzz",
            },
            content_type="application/json",
        )
        self.assertEqual(Session.objects.filter(user=user).count(), 0)


class DemoScenarioFixTests(TestCase):
    """The family_sharing Control Room preset must be time-of-day
    independent: its event timestamp is pinned inside the account's normal
    login window (same replay pattern as patient_attack / genuine_simswap)
    so hour_deviation can never push a legitimate life event into a
    challenge just because the demo is being recorded out-of-window."""

    def test_family_preset_timestamp_pinned_in_window(self):
        from datetime import timedelta

        from django.utils.dateparse import parse_datetime

        from core.demo_scenarios import get_preset_scenarios

        user = make_user(
            typical_login_hours=[[8, 22]],
            typical_transfer_min=1000,
            typical_transfer_max=500000,
            typical_recipients=["BNF-0001"],
            channel_preference=BankUser.CHANNEL_APP,
        )
        make_session(user, timezone.now() - timedelta(days=1),
                     device="dev-A", sim="sim-1", geo="s1tstzz",
                     channel="app")

        scenarios = get_preset_scenarios()
        fam = scenarios["family_sharing"]
        ts = parse_datetime(fam["payload"]["timestamp"])
        self.assertIsNotNone(ts)
        self.assertTrue(
            any(s <= ts.hour < e for s, e in user.typical_login_hours),
            "family_sharing timestamp must fall inside the account's "
            "normal login window so the verdict never depends on the "
            "wall-clock demo hour",
        )


class SecondaryBehaviorProfileTests(TestCase):
    """The verified-secondary-behaviour pattern: a bounded, idempotent,
    customer-isolated profile that can only make sense of the challenges it
    is allowed to touch (confirm / escalate challenge -> block; never touches
    approve or block)."""

    BASE = None

    def _profile(self, user):
        """Fresh DB read of a user's profile (the reverse OneToOne accessor
        caches badly after get_or_create, so never trust it for assertions)."""
        from core.models import SecondaryBehaviorProfile
        return SecondaryBehaviorProfile.objects.filter(user=user).first()

    def _verified_user(self, n=3, amount=25000.0, recipient="BNF-0001",
                       device="dev-A", sim="sim-1", geo="s1tstzz"):
        """A user with ``n`` successful verifications, each a distinct
        session sharing the same verified context."""
        from core.secondary_behavior import record_verification
        user = make_user(typical_recipients=[recipient, "BNF-0002"])
        self.BASE = timezone.now().replace(minute=27, second=0, microsecond=0)
        for i in range(n):
            ts = self.BASE - timedelta(minutes=10 * (n - i))
            s = make_session(user, ts, device=device, sim=sim, geo=geo)
            Transaction.objects.create(
                session=s, timestamp=ts, amount=amount,
                recipient_id=recipient, is_new_recipient=False,
            )
            record_verification(s)
        return user

    def _challenge_decision(self, session):
        return HybridDecision(
            session_id=str(session.session_id),
            score=45,
            verdict="challenge",
            triggered_reasons=[{"code": "context_normal_override", "weight": 0}],
        )

    def test_dormant_below_three_verifications(self):
        from core.secondary_behavior import (
            MIN_VERIFICATIONS_TO_CONFIGURE, apply_secondary_behavior,
        )
        user = self._verified_user(n=2)
        profile = self._profile(user)
        self.assertEqual(profile.verification_count, 2)
        self.assertLess(profile.verification_count,
                        MIN_VERIFICATIONS_TO_CONFIGURE)
        s = make_session(user, self.BASE, device="dev-ZZZ", sim="SIM-ZZZ",
                         geo="s2fffff")
        d = apply_secondary_behavior(s, self._challenge_decision(s))
        # Dormant: verdict untouched, no secondary action taken.
        self.assertEqual(d.verdict, "challenge")
        self.assertFalse(d.secondary_profile_configured)
        self.assertEqual(d.secondary_verification_count, 2)
        self.assertIsNone(d.secondary_action)
        self.assertNotIn("secondary_behavior_escalation",
                         [r["code"] for r in d.triggered_reasons])

    def test_configured_at_three_verifications(self):
        from core.secondary_behavior import (
            MIN_VERIFICATIONS_TO_CONFIGURE,
        )
        user = self._verified_user(n=3)
        self.assertEqual(self._profile(user).verification_count,
                         MIN_VERIFICATIONS_TO_CONFIGURE)

    def test_escalate_challenge_matching_no_verified_behaviour(self):
        from core.secondary_behavior import apply_secondary_behavior
        user = self._verified_user(n=3)
        # A challenged session whose entire context is alien to everything
        # this customer has ever OTP-verified.
        s = make_session(user, self.BASE + timedelta(hours=12),
                         device="dev-ZZZ", sim="SIM-ZZZ", geo="s2fffff")
        Transaction.objects.create(session=s, timestamp=s.timestamp,
                                   amount=99999, recipient_id="BNF-9999",
                                   is_new_recipient=True)
        d = apply_secondary_behavior(s, self._challenge_decision(s))
        self.assertTrue(d.secondary_profile_configured)
        self.assertEqual(d.secondary_action, "escalate")
        self.assertEqual(d.verdict, "block")
        self.assertLess(d.secondary_familiarity,
                        ESCALATE_IF_FAMILIARITY_BELOW)
        self.assertIn("secondary_behavior_escalation",
                      [r["code"] for r in d.triggered_reasons])

    def test_confirm_challenge_matching_verified_behaviour(self):
        from core.secondary_behavior import (
            CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE, apply_secondary_behavior,
        )
        user = self._verified_user(n=3)
        # The exact verified context: same device/SIM/location/recipient,
        # same hour, same amount.
        ts = self.BASE + timedelta(minutes=5)
        s = make_session(user, ts, device="dev-A", sim="sim-1", geo="s1tstzz")
        Transaction.objects.create(session=s, timestamp=ts, amount=25000,
                                   recipient_id="BNF-0001",
                                   is_new_recipient=False)
        d = apply_secondary_behavior(s, self._challenge_decision(s))
        self.assertTrue(d.secondary_profile_configured)
        self.assertEqual(d.secondary_action, "confirm")
        self.assertEqual(d.verdict, "challenge")  # confirmed, not blocked
        self.assertGreaterEqual(d.secondary_familiarity,
                                CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE)
        self.assertIn("secondary_behavior_confirm",
                      [r["code"] for r in d.triggered_reasons])

    def test_mid_range_familiarity_left_alone(self):
        from core.secondary_behavior import (
            CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE,
            ESCALATE_IF_FAMILIARITY_BELOW, apply_secondary_behavior,
        )
        user = self._verified_user(n=3)
        # device + SIM match the verified window (weights 0.25 + 0.20 = 0.45)
        # but location, recipient, hour and amount are entirely alien
        # (0.0 on the remaining 0.55 weight) -> familiarity 0.45, strictly
        # between the thresholds (dormant middle ground: no escalation, no
        # confirmation either -- the challenge is left exactly as-is).
        ts = self.BASE + timedelta(hours=12)
        s = make_session(user, ts, device="dev-A", sim="sim-1", geo="s2fffff")
        Transaction.objects.create(session=s, timestamp=ts, amount=99999,
                                   recipient_id="BNF-9999",
                                   is_new_recipient=False)
        d = apply_secondary_behavior(s, self._challenge_decision(s))
        self.assertEqual(d.secondary_action, "none")
        self.assertEqual(d.verdict, "challenge")
        self.assertGreaterEqual(d.secondary_familiarity,
                                ESCALATE_IF_FAMILIARITY_BELOW)
        self.assertLess(d.secondary_familiarity,
                        CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE)
        self.assertNotIn("secondary_behavior_escalation",
                         [r["code"] for r in d.triggered_reasons])

    def test_approve_and_block_verdicts_never_touched(self):
        from core.secondary_behavior import apply_secondary_behavior
        user = self._verified_user(n=3)
        s = make_session(user, self.BASE, device="dev-ZZZ", sim="SIM-ZZZ",
                         geo="s2fffff")
        # approve must sail through untouched even with a configured profile.
        approve = HybridDecision(
            session_id=str(s.session_id), score=5, verdict="approve",
            triggered_reasons=[],
        )
        out = apply_secondary_behavior(s, approve)
        self.assertEqual(out.verdict, "approve")
        self.assertEqual(out.triggered_reasons, [])
        self.assertIsNone(getattr(out, "secondary_action", None))
        # block is final here: the secondary stage never softens it.
        block = HybridDecision(
            session_id=str(s.session_id), score=80, verdict="block",
            triggered_reasons=[{"code": "impossible_travel", "weight": 50}],
        )
        out2 = apply_secondary_behavior(s, block)
        self.assertEqual(out2.verdict, "block")
        self.assertIsNone(getattr(out2, "secondary_action", None))

    def test_customer_isolation_between_profiles(self):
        from core.secondary_behavior import apply_secondary_behavior
        account_holder = self._verified_user(n=3)
        # The alternate legitimate user verified its OWN (different) behaviour.
        alternate_user = self._verified_user(
            n=3, recipient="BNF-0066",
            device="dev-B", sim="SIM-B", geo="s3aaaaa",
        )
        # The alternate legitimate user never verified dev-A: a challenge on
        # dev-A must escalate against their profile even though the original
        # account holder knows dev-A well. One profile's knowledge never leaks
        # into another.
        ts = self.BASE + timedelta(minutes=7)
        s_b = make_session(alternate_user, ts, device="dev-A", sim="sim-1",
                           geo="s1tstzz")
        Transaction.objects.create(session=s_b, timestamp=ts, amount=25000,
                                   recipient_id="BNF-0001",
                                   is_new_recipient=True)
        d = apply_secondary_behavior(s_b, self._challenge_decision(s_b))
        self.assertEqual(d.verdict, "block")
        self.assertEqual(d.secondary_action, "escalate")
        # The original account holder's profile is untouched by the alternate
        # legitimate user's session.
        self.assertEqual(self._profile(account_holder).verification_count, 3)
        self.assertEqual(self._profile(alternate_user).verification_count, 3)

    def test_verified_window_is_fifo_not_a_whitelist(self):
        from core.secondary_behavior import (
            VERIFIED_HISTORY_LIMIT, record_verification,
        )
        user = make_user()
        base = timezone.now().replace(minute=11, second=0, microsecond=0)
        n = VERIFIED_HISTORY_LIMIT + 3
        for i in range(n):
            ts = base - timedelta(minutes=10 * (n - i))
            s = make_session(user, ts, device=f"dev-{i}", sim=f"SIM-{i}",
                             geo=f"s1tst{i}")
            Transaction.objects.create(session=s, timestamp=ts, amount=1000,
                                       recipient_id=f"BNF-{i:04d}",
                                       is_new_recipient=False)
            record_verification(s)
        profile = self._profile(user)
        ctx = profile.verified_contexts
        self.assertEqual(len(ctx), VERIFIED_HISTORY_LIMIT)
        # The OLDEST verified contexts are gone: no permanent whitelist.
        self.assertNotIn("dev-0", [c["device"] for c in ctx])
        self.assertEqual(ctx[-1]["device"], f"dev-{n - 1}")

    def test_record_verification_is_idempotent(self):
        from core.models import SecondaryVerification
        from core.secondary_behavior import record_verification
        user = make_user()
        s = make_session(user, timezone.now(), device="dev-A", sim="sim-1",
                         geo="s1tstzz")
        Transaction.objects.create(session=s, timestamp=s.timestamp,
                                   amount=1000, recipient_id="BNF-0001",
                                   is_new_recipient=False)
        created1, profile = record_verification(s)
        created2, profile2 = record_verification(s)
        self.assertTrue(created1)
        self.assertFalse(created2)
        self.assertEqual(SecondaryVerification.objects.filter(session=s).count(), 1)
        self.assertEqual(profile.verification_count, 1)
        self.assertEqual(profile2.verification_count, 1)
        self.assertEqual(len(profile.verified_contexts), 1)  # no duplicate

    def test_hold_release_records_verification_idempotently(self):
        from core.bank_views import CHALLENGE_HOLDS
        from core.models import SecondaryVerification
        user = make_user()
        now = timezone.now()
        s = make_session(user, now, device="dev-A", sim="sim-1", geo="s1tstzz")
        txn = Transaction.objects.create(session=s, timestamp=now, amount=1000,
                                         recipient_id="BNF-0001",
                                         is_new_recipient=False)
        ref = "SG-TEST1234"
        CHALLENGE_HOLDS[ref] = {
            "user_id": str(user.user_id),
            "transaction_pk": txn.pk,
            "amount": "1000.00",
            "recipient": "BNF-0001",
        }
        res1 = self.client.post(
            "/api/bank/send-money/",
            {"user_id": str(user.user_id), "challenge_reference": ref},
            content_type="application/json",
        )
        self.assertEqual(res1.status_code, 200)
        body1 = res1.json()
        self.assertEqual(body1["verdict"], "approve")
        self.assertEqual(body1["secondary_verification_count"], 1)
        self.assertFalse(body1["secondary_profile_configured"])
        # A replayed release of the SAME hold is expired (hold already
        # consumed) and must not re-increment the profile.
        res2 = self.client.post(
            "/api/bank/send-money/",
            {"user_id": str(user.user_id), "challenge_reference": ref},
            content_type="application/json",
        )
        self.assertEqual(res2.json()["verdict"], "expired")
        self.assertEqual(SecondaryVerification.objects.filter(session=s).count(), 1)
        self.assertEqual(self._profile(user).verification_count, 1)

    def test_session_event_response_includes_secondary_fields(self):
        user = make_user()
        now = timezone.now()
        res = self.client.post(
            "/api/session-event/",
            {
                "user_id": str(user.user_id),
                "device_fingerprint": "dev-A",
                "sim_id": "sim-1",
                "ip_or_cell_tower_id": "41.2.3.4",
                "location_geohash": "s1tstzz",
                "session_duration_seconds": 60,
                "transaction": {"amount": "25000.00", "recipient_id": "BNF-0001"},
            },
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["verdict"], "approve")
        # Secondary fields present with dormant defaults for a fresh account.
        self.assertFalse(body["secondary_profile_configured"])
        self.assertEqual(body["secondary_verification_count"], 0)
        self.assertIsNone(body["secondary_familiarity"])
        self.assertIsNone(body["secondary_action"])


class SecondaryKeystrokeBehaviorTests(TestCase):
    """The verified-secondary profile also learns typing rhythm from OTP-
    verified APP sessions. Keystroke similarity works as a CAP: a very
    different rhythm (an alternate legitimate user or an impostor) can never
    be CONFIRMED by an otherwise fully verified device/SIM/location context.
    Sessions without typing data keep the pre-keystroke behaviour exactly.
    """

    def setUp(self):
        self.BASE = timezone.now().replace(minute=27, second=0, microsecond=0)

    def _verified_typist(self, n=3):
        from core.secondary_behavior import record_verification
        user = make_user(typical_recipients=["BNF-0001"])
        holds = (150.0, 155.0, 148.0)
        intervals = (1000.0, 990.0, 1010.0)
        cpm = (55.0, 54.0, 56.0)
        hold_stds = (25.0, 26.0, 24.0)
        interval_stds = (60.0, 61.0, 59.0)
        pauses = (1200.0, 1210.0, 1190.0)
        backspaces = (0, 1, 0)
        for i in range(n):
            ts = self.BASE - timedelta(minutes=10 * (n - i))
            s = make_session(user, ts, device="dev-A", sim="sim-1",
                             geo="s1tstzz")
            Transaction.objects.create(
                session=s, timestamp=ts, amount=25000.0,
                recipient_id="BNF-0001", is_new_recipient=False,
            )
            make_keystroke(
                s, hold=holds[i % len(holds)],
                interval=intervals[i % len(intervals)],
                cpm=cpm[i % len(cpm)],
                hold_std=hold_stds[i % len(hold_stds)],
                interval_std=interval_stds[i % len(interval_stds)],
                longest_pause=pauses[i % len(pauses)],
                backspaces=backspaces[i % len(backspaces)],
            )
            record_verification(s)
        return user

    def _challenge_decision(self, session):
        return HybridDecision(
            session_id=str(session.session_id),
            score=45,
            verdict="challenge",
            triggered_reasons=[{"code": "context_normal_override", "weight": 0}],
        )

    def _profile(self, user):
        from core.models import SecondaryBehaviorProfile
        return SecondaryBehaviorProfile.objects.get(user=user)

    def _owner_challenge(self, user, **keystroke_kwargs):
        """A challenge session presenting the fully verified context."""
        ts = self.BASE + timedelta(minutes=5)
        s = make_session(user, ts, device="dev-A", sim="sim-1", geo="s1tstzz")
        Transaction.objects.create(
            session=s, timestamp=ts, amount=25000.0,
            recipient_id="BNF-0001", is_new_recipient=False,
        )
        make_keystroke(s, **keystroke_kwargs)
        return s

    def test_dormant_below_three_even_with_matching_typing(self):
        from core.secondary_behavior import apply_secondary_behavior
        user = self._verified_typist(n=2)
        s = self._owner_challenge(user, hold=152.0, interval=1002.0, cpm=55.0)
        d = apply_secondary_behavior(s, self._challenge_decision(s))
        # Gate still dormant at 2 verifications: no action even though the
        # typing rhythm and context match the original account holder.
        self.assertFalse(d.secondary_profile_configured)
        self.assertEqual(d.secondary_verification_count, 2)
        self.assertIsNone(d.secondary_action)
        self.assertEqual(d.verdict, "challenge")

    def test_activates_at_three_and_confirms_original_holder_rhythm(self):
        from core.secondary_behavior import (
            CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE, apply_secondary_behavior,
        )
        user = self._verified_typist(n=3)
        s = self._owner_challenge(user, hold=152.0, interval=1002.0, cpm=55.0,
                                  hold_std=25.0, interval_std=60.0,
                                  longest_pause=1205.0, backspaces=0)
        d = apply_secondary_behavior(s, self._challenge_decision(s))
        self.assertTrue(d.secondary_profile_configured)
        self.assertEqual(d.secondary_verification_count, 3)
        self.assertEqual(d.secondary_action, "confirm")
        self.assertEqual(d.verdict, "challenge")  # confirmed, not blocked
        self.assertGreaterEqual(d.secondary_familiarity,
                                CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE)
        self.assertIn("secondary_behavior_confirm",
                      [r["code"] for r in d.triggered_reasons])

    def test_matching_rhythm_scores_as_original_account_holder(self):
        from core.secondary_behavior import (
            CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE, _keystroke_features,
            _keystroke_similarity,
        )
        user = self._verified_typist(n=3)
        profile = self._profile(user)
        s = self._owner_challenge(user, hold=152.0, interval=1005.0, cpm=55.0,
                                  hold_std=25.0, interval_std=61.0,
                                  longest_pause=1200.0, backspaces=0)
        sim = _keystroke_similarity(_keystroke_features(s),
                                    profile.verified_contexts)
        self.assertIsNotNone(sim)
        self.assertGreaterEqual(sim, CONFIRM_IF_FAMILIARITY_AT_OR_ABOVE)

    def test_no_accidental_trust_for_a_very_different_rhythm(self):
        from core.secondary_behavior import (
            ESCALATE_IF_FAMILIARITY_BELOW, apply_secondary_behavior,
        )
        user = self._verified_typist(n=3)
        # An alternate legitimate user on the original account holder's
        # rooted device/SIM at the usual location/recipient/hour/amount --
        # the ONLY difference is a completely foreign typing rhythm (slow,
        # clipped, erratic, lots of backspaces).
        s = self._owner_challenge(user, hold=90.0, interval=2500.0, cpm=25.0,
                                  hold_std=250.0, interval_std=900.0,
                                  longest_pause=4000.0, backspaces=6)
        d = apply_secondary_behavior(s, self._challenge_decision(s))
        self.assertTrue(d.secondary_profile_configured)
        # The foreign rhythm CAPS the (otherwise fully familiar) context score
        # below the escalation threshold: never CONFIRMED from context alone.
        self.assertEqual(d.secondary_action, "escalate")
        self.assertEqual(d.verdict, "block")
        self.assertLess(d.secondary_familiarity,
                        ESCALATE_IF_FAMILIARITY_BELOW)
        self.assertIn("secondary_behavior_escalation",
                      [r["code"] for r in d.triggered_reasons])

    def test_sessions_without_typing_data_keep_pre_keystroke_behaviour(self):
        from core.secondary_behavior import (
            apply_secondary_behavior, record_verification,
        )
        # Verified window WITHOUT keystroke rows -> no keystroke cap.
        user = make_user(typical_recipients=["BNF-0001"])
        for i in range(3):
            ts = self.BASE - timedelta(minutes=10 * (3 - i))
            s = make_session(user, ts, device="dev-A", sim="sim-1",
                             geo="s1tstzz")
            Transaction.objects.create(
                session=s, timestamp=ts, amount=25000.0,
                recipient_id="BNF-0001", is_new_recipient=False,
            )
            record_verification(s)
        ts = self.BASE + timedelta(minutes=5)
        s = make_session(user, ts, device="dev-A", sim="sim-1", geo="s1tstzz")
        Transaction.objects.create(session=s, timestamp=ts, amount=25000,
                                   recipient_id="BNF-0001",
                                   is_new_recipient=False)
        d = apply_secondary_behavior(s, self._challenge_decision(s))
        # Full six-dimension context match -> confirmed, exactly as before
        # the keystroke feature existed.
        self.assertEqual(d.secondary_action, "confirm")
        self.assertEqual(d.verdict, "challenge")

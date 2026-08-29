"""
Real-time scoring API -- the production tie-together.

One view class serves both endpoints; channel is bound per-URL via
``as_view(channel="app")`` / ``as_view(channel="ussd")`` rather than two
diverging implementations. Rationale: with one purpose and one payload
shape, two thin URL bindings onto ONE validated pipeline guarantee app and
USSD events can never drift apart in behaviour -- same single-source
principle as the shared feature engine.

Example request (both endpoints; channel is forced by which URL you hit):
{
    "user_id": "<existing BankUser uuid>",
    "device_fingerprint": "<64-hex hash>",   # app only; ignored on ussd
    "sim_id": "<64-hex hash>",
    "ip_or_cell_tower_id": "41.2.3.4",       # or "TWR-ab12cd34"
    "location_geohash": "s1tst0",
    "session_duration_seconds": 95,
    "transaction": {"amount": "25000.00", "recipient_id": "BNF-ABC123"}
}

Response 200:
{
    "session_id": "...",
    "verdict": "challenge",          # approve | challenge | block
    "score": 88,
    "customer_message": "We asked for extra verification ...",
    "internal_note": "Note: hardware change but otherwise normal ...",
    "debug_signals": [{"code": "ml_model_risk", "weight": 88}],
    "ml_probability": 0.88,
    "context_override_applied": true,
    "warnings": []
}

``customer_message`` is what a bank tells the customer.
``internal_note``/``debug_signals`` are for logs and demos only -- a real
production API would not expose them to end users.

Errors: 400 invalid input | 404 unknown user_id | 500 generic body with
the real exception logged server-side.
"""

import logging

from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from core.explanation import explain_decision
from core.feature_engine import compute_features, load_user_history
from core.hybrid_scorer import score_session_hybrid
from core.models import BankUser, KeystrokeDynamics, Session, Transaction
from core.serializers import SessionEventSerializer

logger = logging.getLogger(__name__)

# Warm the trained-model cache at import/startup time so the FIRST real
# request is not billed the ~9s joblib deserialisation cost -- keeps every
# scored event inside the sub-second SLA. (Safe no-op if no bundle yet.)
try:
    from core.ml_model import _load_bundle as _warm_model_cache
    _warm_model_cache()
except Exception:  # pragma: no cover - dev machines before first training
    logger.warning(
        "trained_model.joblib not found at startup; ML scores will load "
        "lazily on first request."
    )


class SessionEventView(APIView):
    """Accept one session event, persist it as history, score + explain."""

    channel = None  # bound per-URL: "app" | "ussd"

    def post(self, request):
        payload = dict(request.data)
        # Endpoint is authoritative on channel: clients cannot post USSD
        # traffic to the app route or vice versa.
        payload["channel"] = self.channel

        serializer = SessionEventSerializer(data=payload)
        if not serializer.is_valid():
            return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)
        data = serializer.validated_data
        warnings = data.pop("warnings", [])

        try:
            user = BankUser.objects.get(user_id=data["user_id"])
        except BankUser.DoesNotExist:
            return Response(
                {"error": f"Unknown bank user {data['user_id']}. Register "
                          f"the customer before sending session events."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            outcome = self._score_event(user, data)
        except Exception:
            logger.exception("Scoring failed for user %s", data["user_id"])
            return Response(
                {"error": "Internal error while scoring this event.",
                 "warnings": warnings},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        outcome["warnings"] = warnings
        return Response(outcome, status=status.HTTP_200_OK)

    def _score_event(self, user, data):
        """Steps c-g of the pipeline; exceptions propagate to post()'s
        catch-all so clients never see stack traces."""
        # Client-supplied event time (store-and-forward replays, demo
        # presets) or server 'now' -- explicit per our auto_now_add fix.
        # Callers that skip the DRF serializer (bank ledger endpoint)
        # hand us a raw ISO string; parse it so the feature engine and
        # Session.objects.create() always see a datetime.
        now = data.get("timestamp")
        if isinstance(now, str):
            now = parse_datetime(now)
        if not now:
            now = timezone.now()

        # History BEFORE this session exists, exactly as the feature engine
        # would load it -- reused for both the new-device/new-sim flags and
        # compute_features, so API events and batch data stay consistent.
        history = load_user_history(user.user_id, before=now)

        fingerprint = data.get("device_fingerprint")
        is_new_device = (
            bool(history.device_counts)
            and bool(fingerprint)
            and fingerprint not in history.device_counts
        )
        dominant_sim = history.dominant_sim()
        is_new_sim = (
            dominant_sim is not None
            and data["sim_id"] != dominant_sim
        )

        session = Session.objects.create(
            user=user,
            timestamp=now,
            channel=data["channel"],
            device_fingerprint=fingerprint,
            sim_id=data["sim_id"],
            ip_or_cell_tower_id=data["ip_or_cell_tower_id"],
            location_geohash=data["location_geohash"],
            session_duration_seconds=data.get("session_duration_seconds"),
            is_new_device=is_new_device,
            is_new_sim=is_new_sim,
        )

        txn_payload = data.get("transaction")
        if txn_payload:
            Transaction.objects.create(
                session=session,
                amount=txn_payload["amount"],
                recipient_id=txn_payload["recipient_id"],
                narration=txn_payload.get("narration", ""),
                is_new_recipient=(
                    txn_payload["recipient_id"]
                    not in user.typical_recipients
                ),
            )

        # Keystroke evidence (app channel) is persisted BEFORE feature
        # computation so it is part of THIS session's feature vector --
        # previously bank_send_money created it AFTER scoring, meaning the
        # very transaction being examined could never see its own pattern.
        # login_pin_failures is the login-phase signal added by this task.
        ks_payload = data.get("keystroke")
        if ks_payload and session.channel == "app":
            try:
                KeystrokeDynamics.objects.create(
                    session=session,
                    avg_hold_time_ms=float(ks_payload.get("avg_hold_time_ms", 0)),
                    avg_interval_ms=float(ks_payload.get("avg_interval_ms", 0)),
                    typing_speed_cpm=float(ks_payload.get("typing_speed_cpm", 0)),
                    login_pin_failures=(
                        int(ks_payload["login_pin_failures"])
                        if ks_payload.get("login_pin_failures") is not None
                        else None
                    ),
                )
            except (ValueError, TypeError):
                logger.warning(
                    "Malformed keystroke payload ignored for session %s",
                    session.session_id,
                )

        features = compute_features(session, history=history)
        features.save()  # a real event now: persisted like training history

        # Unified entry point: normally identical to score_session_hybrid,
        # but during a (real or demo-simulated) outage it degrades to the
        # local rules-only check and spools this event for resync.
        from core.offline_fallback import score_session_with_fallback
        decision = score_session_with_fallback(session, features=features)

        explained = explain_decision(decision)
        if isinstance(explained, tuple):
            customer_message, internal_note = explained
        else:
            customer_message, internal_note = explained, None

        return {
            "session_id": str(session.session_id),
            "verdict": decision.verdict,
            "score": decision.score,
            "customer_message": customer_message,
            "internal_note": internal_note,
            "debug_signals": decision.triggered_reasons,
            # Hybrid-only attributes: absent on degraded decisions, where
            # there IS no ML probability and no override concept.
            "ml_probability": (
                round(decision.ml_probability, 2)
                if getattr(decision, "ml_probability", None) is not None
                else None
            ),
            "context_override_applied": bool(
                getattr(decision, "context_override_applied", False)
            ),
            "is_degraded": bool(getattr(decision, "is_degraded", False)),
        }


def run_scoring_pipeline(user, data):
    """Module-level entry to the SAME pipeline the production endpoints
    use, so the demo bank app / USSD simulator get identical risk
    treatment (and the offline fallback) with zero duplicated logic.
    ``SessionEventView._score_event`` never touches ``self``."""
    return SessionEventView._score_event(None, user, data)


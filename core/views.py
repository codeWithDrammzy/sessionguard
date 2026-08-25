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
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from core.explanation import explain_decision
from core.feature_engine import compute_features, load_user_history
from core.hybrid_scorer import score_session_hybrid
from core.models import BankUser, Session, Transaction
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
        now = timezone.now()  # explicit timestamp per our auto_now_add fix

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
                is_new_recipient=(
                    txn_payload["recipient_id"]
                    not in user.typical_recipients
                ),
            )

        features = compute_features(session, history=history)
        features.save()  # a real event now: persisted like training history

        decision = score_session_hybrid(features)

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
            "ml_probability": round(decision.ml_probability, 2),
            "context_override_applied": decision.context_override_applied,
        }


"""
Demo-only endpoints + page for the live Control Room presentation.

Deliberately SEPARATE from production SessionEventView: these routes exist
to make judging easy (preset scenarios, an outage switch, a dashboard).
None of this would ship in a real deployment -- each piece is commented as
demo-only at the point of use.
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

from rest_framework.decorators import api_view  # noqa: E402
from rest_framework.response import Response  # noqa: E402
from django.views.generic import TemplateView  # noqa: E402

from core.demo_scenarios import get_preset_scenarios  # noqa: E402
from core.models import ConfirmedOutcome, Session  # noqa: E402


@api_view(["GET"])
def demo_scenarios(request):
    """DEMO ONLY: the five preset payloads built from real DB rows."""
    return Response(get_preset_scenarios())


@api_view(["POST"])
def toggle_offline(request):
    """DEMO ONLY: flip the simulated network-outage switch.

    Body: {"offline": true|false}. Returns the resulting state. This is
    process-memory state for presentations -- NOT a production control.
    """
    from core.offline_fallback import (
        _DEMO_FORCED_OFFLINE, set_demo_offline_mode,
    )

    wanted = bool(request.data.get("offline", False))
    state = set_demo_offline_mode(wanted)
    return Response({
        "offline": state,
        "message": ("Network outage SIMULATED -- scoring is now degraded "
                    "and events are queued for resync."
                    if state else
                    "Network restored -- full hybrid scoring active."),
    })


@api_view(["POST"])
def confirm_outcome(request):
    """DEMO ONLY: record a HUMAN-confirmed outcome for a scored session.

    Body: {"session_id": "<uuid>", "confirmed_attack": true|false,
           "confirmed_by": "<optional reviewer note>"}

    This is the "the model keeps learning" hook: after the system returns a
    verdict, a reviewer (judge / analyst) inspects the event and records what
    it ACTUALLY was. ``ConfirmedOutcome`` rows then fold into the next ML
    retrain (see core/ml_model.build_dataset). Writing the same session twice
    upserts (one confirmation per session).
    """
    session_id = (request.data.get("session_id") or "").strip()
    confirmed_attack = bool(request.data.get("confirmed_attack", False))
    confirmed_by = (request.data.get("confirmed_by") or "demo").strip()[:120]

    try:
        session = Session.objects.get(session_id=session_id)
    except (Session.DoesNotExist, ValueError):
        return Response(
            {"error": "Unknown or malformed session_id."},
            status=400,
        )

    obj, created = ConfirmedOutcome.objects.update_or_create(
        session=session,
        defaults={
            "confirmed_attack": confirmed_attack,
            "confirmed_by": confirmed_by,
            "source": "demo",
        },
    )

    n_attack = ConfirmedOutcome.objects.filter(
        confirmed_attack=True
    ).count()
    n_total = ConfirmedOutcome.objects.count()
    return Response({
        "ok": True,
        "created": created,
        "confirmed_attack": obj.confirmed_attack,
        # Confirmed-outcome running totals the judge view can echo.
        "confirmed_total": n_total,
        "confirmed_attacks": n_attack,
        "hint": (
            "Run `python manage.py retrain_model` to fold these confirmed "
            "outcomes into the ML model."
        ),
    })


class ControlRoomView(TemplateView):
    """DEMO ONLY: the single-page Control Room dashboard."""

    template_name = "demo/control_room.html"

    def get(self, request, *args, **kwargs):
        response = super().get(request, *args, **kwargs)
        # Never cache the demo bundle (see BankAppView).
        response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response["Pragma"] = "no-cache"
        response["Expires"] = "0"
        return response

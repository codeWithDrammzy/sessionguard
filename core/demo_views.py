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


class ControlRoomView(TemplateView):
    """DEMO ONLY: the single-page Control Room dashboard."""

    template_name = "demo/control_room.html"

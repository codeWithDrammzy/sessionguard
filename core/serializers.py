"""
DRF input validation for the real-time scoring API.

SessionEventSerializer accepts one login/transaction event and normalises
it into exactly the fields needed to persist a Session (+ optional
Transaction). Channel-specific conventions mirror the data schema:

  * USSD sessions have no device concept -- an incoming device_fingerprint
    on a USSD event is silently dropped (stored NULL), matching the
    null-for-USSD rule every generator and the feature engine rely on.
  * App sessions SHOULD carry a fingerprint, but a missing one only raises
    a non-fatal warning (surfaced to the caller via ``validated_data``
    warnings): early-session edge cases can genuinely lack it, and the
    feature engine already treats absent devices as "nothing to compare".

The BankUser existence check deliberately lives in the VIEW (not here) so
a valid-format but unknown user_id yields a clean 404 rather than DRF's
generic 400 validation shape.
"""

from rest_framework import serializers


class TransactionEventSerializer(serializers.Serializer):
    """Optional nested transfer payload."""

    amount = serializers.DecimalField(
        max_digits=14,
        decimal_places=2,
        min_value=0,
        help_text="Transfer amount in Naira.",
    )
    recipient_id = serializers.CharField(
        max_length=64,
        help_text="Beneficiary identifier for the transfer.",
    )


class SessionEventSerializer(serializers.Serializer):
    """One login/session event arriving for real-time scoring."""

    user_id = serializers.UUIDField(
        help_text="Existing BankUser identifier (unknown ids -> HTTP 404).",
    )
    channel = serializers.ChoiceField(
        choices=["app", "ussd"],
        help_text="Overridden per-endpoint: /api/ussd-event/ forces ussd.",
    )
    device_fingerprint = serializers.CharField(
        max_length=128,
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Hashed device identity; app sessions only.",
    )
    sim_id = serializers.CharField(
        max_length=64,
        help_text="Salted hash of the SIM identity (never raw IMSI).",
    )
    ip_or_cell_tower_id = serializers.CharField(
        max_length=64,
        help_text="Client IP (app) or cell tower ID (USSD).",
    )
    location_geohash = serializers.CharField(
        max_length=12,
        help_text="Coarse (city-level) geohash.",
    )
    session_duration_seconds = serializers.IntegerField(
        required=False,
        allow_null=True,
        min_value=0,
        help_text="How long the session ran; may arrive later/absent.",
    )
    transaction = TransactionEventSerializer(
        required=False,
        allow_null=True,
        help_text="Optional transfer attached to this session.",
    )

    def validate(self, attrs):
        warnings = []
        channel = attrs["channel"]
        fingerprint = attrs.get("device_fingerprint")

        if channel == "ussd":
            # Schema convention: USSD exposes no device identity. Drop it
            # silently -- sending one is not a client error worth failing.
            if fingerprint:
                warnings.append(
                    "device_fingerprint ignored for ussd channel events."
                )
            attrs["device_fingerprint"] = None
        else:  # app
            if not fingerprint:
                # Warn-but-continue: early-session events may lack it; the
                # feature engine handles absence as 'nothing to compare'.
                warnings.append(
                    "device_fingerprint missing on an app session; scored "
                    "without device comparison."
                )
            if fingerprint == "":
                attrs["device_fingerprint"] = None

        attrs["warnings"] = warnings
        return attrs

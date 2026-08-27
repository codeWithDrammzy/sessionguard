"""
Demo banking layer for the customer-facing experience (bank app + USSD
simulator). NOT production code -- it exists so judges can watch real
customers meet the risk engine.

Design notes:
  * ONE ledger endpoint (/api/bank/send-money/) serves BOTH channels. It
    internally calls the very same run_scoring_pipeline as the raw
    /api/session-event/ and /api/ussd-event/ routes, then COMMITS the
    transfer only on "approve". This keeps the ledger consistent across
    channels instead of duplicating deduction logic per route.
  * Challenges place a HOLD: the transfer is not committed and a reference
    is returned. The customer completes SMS-style verification (demo: any
    numeric code) and the hold is released -- mirroring how step-up
    verification releases a held payment at real banks.
  * Blocked transfers are never committed; funds never move.
"""

import os
import sys
import uuid
from collections import OrderedDict

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE", "sessionguard_project.settings"
    )
    import django

    django.setup()

from decimal import Decimal  # noqa: E402

from django.db.models import F  # noqa: E402
from django.utils import timezone  # noqa: E402
from django.views.generic import TemplateView  # noqa: E402
from rest_framework import status as drf_status  # noqa: E402
from rest_framework.decorators import api_view  # noqa: E402
from rest_framework.response import Response  # noqa: E402

from core.models import BankUser, KeystrokeDynamics, RecipientDirectory, Transaction  # noqa: E402
from core.views import run_scoring_pipeline  # noqa: E402

# Holds created by "challenge" verdicts: reference -> context. In-memory
# on purpose (prototype): restarting the server drops unverified holds,
# exactly like an unclaimed OTP expiring. Bounded to stay honest about it.
CHALLENGE_HOLDS = OrderedDict()
HOLD_LIMIT = 200


def _new_reference() -> str:
    return "SG-" + uuid.uuid4().hex[:8].upper()


def _recent_activity(user, limit=8):
    txns = (Transaction.objects
            .filter(session__user=user)
            .exclude(reference="", outcome="approve")  # hide legacy rows
            .order_by("-timestamp")[:limit])
    return [
        {
            "reference": t.reference or "-",
            "amount": str(t.amount),
            "recipient": t.recipient_id,
            "outcome": t.outcome,
            "channel": t.session.channel,
            "time": timezone.localtime(t.timestamp)
                              .strftime("%d %b %H:%M"),
        }
        for t in txns
    ]


@api_view(["POST"])
def bank_signup(request):
    """DEMO: self-service account creation with a permissive starting
    profile -- wide login hours, generous amount band, empty history. The
    feature engine treats no-history as 'nothing to compare' (never
    punished), so brand-new customers are not friction-bombed."""
    first_name = (request.data.get("first_name") or "").strip()
    phone = (request.data.get("phone_number") or "").strip()
    if not first_name or not phone:
        return Response(
            {"error": "first_name and phone_number are required."},
            status=drf_status.HTTP_400_BAD_REQUEST,
        )
    if BankUser.objects.filter(phone_number=phone).exclude(
            phone_number="").exists():
        return Response(
            {"error": "That phone number is already registered."},
            status=drf_status.HTTP_400_BAD_REQUEST,
        )
    user = BankUser.objects.create(
        first_name=first_name,
        phone_number=phone,
        channel_preference="app",
        typical_login_hours=[[0, 24]],  # permissive until history forms
        typical_transfer_min=Decimal("500.00"),
        typical_transfer_max=Decimal("200000.00"),
        typical_recipients=[],
        registered_devices=[],
        account_age_days=0,
    )
    return Response({
        "user_id": str(user.user_id),
        "first_name": user.first_name,
        "balance": str(user.balance),
    }, status=drf_status.HTTP_201_CREATED)


@api_view(["POST"])
def bank_login(request):
    """DEMO: phone-number 'login' (no passwords -- prototype scope).
    Returns the same shape as signup so the frontend treats both alike."""
    phone = (request.data.get("phone_number") or "").strip()
    try:
        user = BankUser.objects.get(phone_number=phone)
    except BankUser.DoesNotExist:
        return Response({"error": "No account found for that number."},
                        status=drf_status.HTTP_404_NOT_FOUND)
    return Response({
        "user_id": str(user.user_id),
        "first_name": user.first_name,
        "balance": str(user.balance),
    })


@api_view(["GET"])
def bank_state(request, user_id):
    """DEMO: dashboard payload -- identity, balance, recent activity."""
    try:
        user = BankUser.objects.get(user_id=user_id)
    except BankUser.DoesNotExist:
        return Response({"error": "Unknown account."},
                        status=drf_status.HTTP_404_NOT_FOUND)
    return Response({
        "user_id": str(user.user_id),
        "first_name": user.first_name,
        "balance": str(user.balance),
        "recent": _recent_activity(user),
    })


@api_view(["POST"])
def bank_send_money(request):
    """DEMO: score a transfer through the REAL pipeline and commit it
    only when the verdict is approve. Challenges create a hold that the
    SMS-verification step can release; blocks never touch the balance.

    Body: {user_id, channel: app|ussd, device_fingerprint, sim_id,
           ip_or_cell_tower_id, location_geohash,
           transaction: {amount, recipient_id}   # omit for balance check
           , challenge_reference?: "<hold ref>"}
    """
    try:
        user = BankUser.objects.get(user_id=request.data.get("user_id"))
    except BankUser.DoesNotExist:
        return Response({"error": "Unknown account."},
                        status=drf_status.HTTP_404_NOT_FOUND)

    # ---- hold-release path (SMS verification succeeded) ----------------
    hold_ref = request.data.get("challenge_reference")
    if hold_ref:
        hold = CHALLENGE_HOLDS.get(hold_ref)
        if not hold or hold["user_id"] != str(user.user_id):
            return Response(
                {"verdict": "expired",
                 "customer_message":
                     "This verification request has expired. Please start "
                     "the transfer again.",
                 "reference": hold_ref},
                status=drf_status.HTTP_200_OK,
            )
        CHALLENGE_HOLDS.pop(hold_ref, None)
        txn = Transaction.objects.filter(pk=hold["transaction_pk"]).first()
        amount = Decimal(hold["amount"])
        BankUser.objects.filter(pk=user.pk).update(
            balance=F("balance") - amount)
        user.refresh_from_db()
        if txn:
            txn.outcome = Transaction.OUTCOME_APPROVE
            txn.save()
        return Response({
            "verdict": "approve",
            "released_hold": True,
            "reference": hold_ref,
            "customer_message":
                f"Verification successful. Your transfer of NGN "
                f"{amount:,.2f} to {hold['recipient']} is complete.",
            "balance": str(user.balance),
        })

    # ---- normal scoring path -------------------------------------------
    data = {
        "user_id": str(user.user_id),
        "channel": request.data.get("channel", "app"),
        "device_fingerprint": request.data.get("device_fingerprint"),
        "sim_id": request.data.get("sim_id", ""),
        "ip_or_cell_tower_id": request.data.get("ip_or_cell_tower_id", ""),
        "location_geohash": request.data.get("location_geohash", ""),
        "session_duration_seconds": request.data.get(
            "session_duration_seconds"),
        # Forward client event time (store-and-forward replays); the
        # serializer validates it and _score_event falls back to now().
        "timestamp": request.data.get("timestamp"),
    }
    txn_payload = request.data.get("transaction")
    if txn_payload:
        data["transaction"] = {
            "amount": str(txn_payload["amount"]),
            "recipient_id": txn_payload["recipient_id"],
            "narration": txn_payload.get("narration", ""),
        }

    result = run_scoring_pipeline(user, data)
    verdict = result["verdict"]
    txn = Transaction.objects.filter(
        session_id=result["session_id"]).first()

    # --- Keystroke dynamics: aggregate timing from the full user journey ----
    ks = request.data.get("keystroke")
    if ks and result.get("session_id"):
        try:
            from core.models import Session as _S
            session_obj = _S.objects.get(session_id=result["session_id"])
            KeystrokeDynamics.objects.create(
                session=session_obj,
                avg_hold_time_ms=float(ks.get("avg_hold_time_ms", 0)),
                avg_interval_ms=float(ks.get("avg_interval_ms", 0)),
                typing_speed_cpm=float(ks.get("typing_speed_cpm", 0)),
            )
        except Exception:
            pass  # best-effort: don't fail the transfer over analytics

    if txn and txn_payload:
        if verdict == "approve":
            ref = _new_reference()
            txn.reference = ref
            txn.outcome = Transaction.OUTCOME_APPROVE
            txn.save()
            BankUser.objects.filter(pk=user.pk).update(
                balance=F("balance") - Decimal(str(txn.amount)))
            user.refresh_from_db()
            result["reference"] = ref
            result["balance"] = str(user.balance)
        elif verdict == "challenge":
            ref = _new_reference()
            txn.reference = ref
            txn.outcome = Transaction.OUTCOME_CHALLENGE
            txn.save()
            while len(CHALLENGE_HOLDS) >= HOLD_LIMIT:
                CHALLENGE_HOLDS.popitem(last=False)
            CHALLENGE_HOLDS[ref] = {
                "user_id": str(user.user_id),
                "transaction_pk": txn.pk,
                "amount": str(txn.amount),
                "recipient": txn.recipient_id,
            }
            result["reference"] = ref
            result["hold"] = True
        else:  # block
            txn.reference = _new_reference()
            txn.outcome = Transaction.OUTCOME_BLOCK
            txn.save()
            result["reference"] = txn.reference
    elif not txn_payload and verdict == "approve":
        result["balance"] = str(user.balance)  # balance-check session

    return Response(result, status=drf_status.HTTP_200_OK)


# --- Nigerian name pool for realistic account-name lookup ---
_NG_FIRST = [
    "Adaeze", "Chidinma", "Emeka", "Fatima", "Ibrahim",
    "Kemi", "Ngozi", "Obinna", "Olumide", "Sade",
    "Tunde", "Uche", "Yemi", "Chukwuemeka", "Amina",
    "Babatunde", "Funke", "Grace", "Hauwa", "Ifeanyi",
]
_NG_LAST = [
    "Nwosu", "Okonkwo", "Okafor", "Adeyemi", "Bello",
    "Ogundimu", "Chukwu", "Abubakar", "Eze", "Onwueme",
    "Oladipo", "Nnamdi", "Bankole", "Mohammed", "Akinwale",
    "Oyewole", "Igwe", "Suleiman", "Adebanjo", "Obi",
]


@api_view(["POST"])
def bank_lookup_account(request):
    """DEMO: Nigerian banking 'name enquiry' -- given a 10-digit NUBAN
    account number, return the account holder's display name.

    If the account is already in RecipientDirectory, return the stored
    name (consistent across transfers). Otherwise, generate a plausible
    random Nigerian full name, persist it, and return it.
    """
    acct = (request.data.get("account_number") or "").strip()
    if not acct.isdigit() or len(acct) != 10:
        return Response(
            {"error": "account_number must be exactly 10 digits."},
            status=drf_status.HTTP_400_BAD_REQUEST,
        )
    import random as _rng
    entry, _ = RecipientDirectory.objects.get_or_create(
        account_number=acct,
        defaults={
            "display_name": (
                f"{_rng.choice(_NG_FIRST)} {_rng.choice(_NG_LAST)}"
            ).upper(),
        },
    )
    return Response({
        "account_number": acct,
        "display_name": entry.display_name,
    })


class BankAppView(TemplateView):
    """DEMO: the customer-facing experience page (app + USSD simulator)."""

    template_name = "bank/bank_app.html"

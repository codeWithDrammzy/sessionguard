"""Reset the demo database to its known-good snapshot.

Deletes ANY test/debris rows so a fresh demo starts from exactly the 250
seeded synthetic customers, with a pristine offline resync queue. This is a
DEMO-HYGIENE command: it never touches labelled (FraudLabel) rows, which are
the synthetic training/evaluation ground truth owned by the dataset
generators.

What it removes (each documented in core/):
* Browser-created test accounts -- BankUser rows with a phone_number set.
  The 250 seeded customers all have empty phone/first_name; only a visitor
  who actually signed up through the live app carries a phone number. Deleting
  the user cascades to their sessions, transactions, features and keystrokes.
* Demo Control-Room debris -- sessions stamped with a reserved debris marker
  (DEMO_TOWER / SMOKE_TOWER / OFFLINE_TOWER, shared with demo_scenarios).
* Smoke-test debris -- sessions stamped SMOKE_TOWER (same whitelist).
* offline_queue.jsonl -- stale resync entries from an offline demo run, so a
  judge never sees phantom queued traffic.

Run:
    python manage.py reset_demo [--check]
    --check prints what WOULD be removed without deleting anything.
"""
import os
import sys

from django.core.management.base import BaseCommand

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
QUEUE_PATH = os.path.join(PROJECT_ROOT, "offline_queue.jsonl")


class Command(BaseCommand):
    help = "Reset the demo DB to the seeded 250-customer snapshot."

    def add_arguments(self, parser):
        parser.add_argument(
            "--check",
            action="store_true",
            help="Report what would be removed without deleting anything.",
        )

    def handle(self, *args, **options):
        check = options["check"]

        from core.demo_scenarios import DEBRIS_TOWERS
        from core.models import BankUser, Session, BehavioralFeatures

        # --- 1. Browser-created test accounts ---------------------------------
        test_users = list(
            BankUser.objects.exclude(phone_number="").order_by("user_id")
        )
        # Any orphan feature/session rows whose user was already removed
        # (defensive, in case a previous partial wipe left orphans).
        orphan_features = BehavioralFeatures.objects.filter(
            session__user__isnull=True
        )
        orphan_sessions = Session.objects.filter(user__isnull=True)

        # --- 2. Demo + smoke debris (marker whitelist only) -------------------
        # Exactly the same rule as core.demo_scenarios._purge_previous_demo_traffic:
        # a session is debris if and only if its tower ID is a reserved marker.
        # There is NO same-day-unlabelled heuristic: the seeded baseline
        # legitimately has sessions dated "today", and the old sweep deleted
        # those real rows (measured at 84 on one Control Room load).
        marked = Session.objects.filter(ip_or_cell_tower_id__in=DEBRIS_TOWERS)

        counts = {
            "test_accounts": len(test_users),
            "orphan_feature_rows": orphan_features.count(),
            "orphan_session_rows": orphan_sessions.count(),
            "demo_marker_sessions": marked.count(),
            "offline_queue_entries": (
                sum(1 for _ in open(QUEUE_PATH, "r", encoding="utf-8"))
                if os.path.exists(QUEUE_PATH)
                else 0
            ),
        }

        self.stdout.write("DEMO RESET SUMMARY")
        for k, v in counts.items():
            self.stdout.write(f"  {k:<28} : {v}")

        if check:
            self.stdout.write(self.style.WARNING("--check: nothing deleted."))
            return

        if test_users:
            BankUser.objects.filter(pk__in=[u.pk for u in test_users]).delete()
        orphan_sessions.delete()
        orphan_features.delete()
        marked.delete()

        # --- 3. Clear the offline resync queue --------------------------------
        if os.path.exists(QUEUE_PATH):
            open(QUEUE_PATH, "w", encoding="utf-8").close()
            self.stdout.write("  offline_queue.jsonl cleared.")

        users = BankUser.objects.count()
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. {users} customers (250 seeded + "
                f"{users - 250} remaining test). Sessions="
                f"{Session.objects.count()}, "
                f"Txns="
                f"{_txn_count()}."
            )
        )


def _txn_count():
    from core.models import Transaction
    return Transaction.objects.count()

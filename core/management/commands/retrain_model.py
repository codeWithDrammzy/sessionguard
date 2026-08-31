"""Re-run the ML training pipeline, folding in human-confirmed outcomes.

This is the operational half of "the model keeps learning": it re-reads the
database (synthetic FraudLabel ground truth PLUS any ConfirmedOutcome rows a
reviewer/judge recorded in the demo), retrains the interpretable
LogisticRegression, prints the fresh evaluation, and saves the updated
bundle. A running server picks it up on the next request via
``core.ml_model.reload_bundle`` (no process restart needed).

Run:
    python manage.py retrain_model [--no-reload]
    --no-reload skips clearing the in-process bundle cache (useful if the
               retrain is happening from a separate process that is NOT the
               serving process, e.g. a one-off cron-style run).
"""
import os
import sys

from django.core.management.base import BaseCommand
from django.utils import timezone

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, PROJECT_ROOT)


class Command(BaseCommand):
    help = "Retrain the ML model, folding confirmed demo outcomes into the data."

    def add_arguments(self, parser):
        parser.add_argument(
            "--no-reload",
            action="store_true",
            help="Do not clear the in-process bundle cache after saving.",
        )

    def handle(self, *args, **options):
        from core.ml_model import train_and_evaluate, reload_bundle

        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"Retraining ML model at {timezone.now().isoformat()}"
            )
        )
        train_and_evaluate()

        if not options["no_reload"]:
            reload_bundle()
            self.stdout.write(
                self.style.SUCCESS(
                    "Bundle cache cleared -- a running server adopts the "
                    "new model on its next scoring request."
                )
            )

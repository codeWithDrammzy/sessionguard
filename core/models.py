"""
SessionGuard data models
========================

Domain model for a real-time behavioural account-takeover (ATO) detection
system aimed at Nigerian banking channels (mobile app + USSD).

The modelling philosophy:

* ``BankUser``     -- the bank account being protected, plus the behavioural
                      "baseline" we learn about it (normal hours, normal
                      amounts, known recipients/devices). This is intentionally
                      NOT Django's auth user: it models an account holder in
                      the bank's core-banking world, not an admin login.
* ``Session``      -- one login/interaction on either channel (app or USSD),
                      with the raw context signals available at login time.
* ``Transaction``  -- a money-transfer attempt that happened inside a session.
* ``BehavioralFeatures`` -- the engineered feature vector computed at scoring
                      time (one-to-one with a Session) and fed to the risk
                      model.
* ``FraudLabel``   -- ground truth for training/evaluation on the synthetic
                      dataset only; never consulted in the production scoring
                      path.

All primary keys are UUIDs so identifiers are non-guessable and safe to expose
in APIs/logs without leaking production volume.
"""

import uuid

from django.db import models
from django.utils import timezone


class BankUser(models.Model):
    """
    A bank account under protection, together with its learned behavioural
    baseline.

    Named ``BankUser`` (not ``User``) to avoid clashing with
    ``django.contrib.auth.models.User`` / ``AUTH_USER_MODEL``, which this
    project deliberately does not use for account holders.

    The ``typical_*`` fields form the per-customer profile that deviation
    scores are computed against at scoring time.
    """

    # Choices for the channel(s) the customer is known to use.
    CHANNEL_APP = "app"
    CHANNEL_USSD = "ussd"
    CHANNEL_BOTH = "both"
    CHANNEL_PREFERENCE_CHOICES = [
        (CHANNEL_APP, "Mobile app"),
        (CHANNEL_USSD, "USSD"),
        (CHANNEL_BOTH, "Both app and USSD"),
    ]

    # Non-guessable public identifier for the account.
    user_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique identifier for the bank account being protected.",
    )

    # List of hour ranges (e.g. [[7, 9], [18, 22]]) in which the user normally
    # logs in. Used to compute the hour-deviation feature at scoring time.
    typical_login_hours = models.JSONField(
        default=list,
        blank=True,
        help_text="Hour ranges (0-23) during which this user normally logs in.",
    )

    # Bounds of the user's historical transfer behaviour. Amounts outside
    # these bounds contribute to the amount-deviation score.
    typical_transfer_min = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Smallest transfer amount typically made by this user.",
    )
    typical_transfer_max = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Largest transfer amount typically made by this user.",
    )

    # Recipient IDs (account numbers/beneficiary IDs) seen in past transfers.
    # A recipient not on this list raises the new-recipient flag.
    typical_recipients = models.JSONField(
        default=list,
        blank=True,
        help_text="List of beneficiary IDs this user has transferred to before.",
    )

    # Device fingerprints observed on past logins from this account. A
    # fingerprint not on this list marks the session as coming from a new
    # device (relevant for the device-change flag).
    registered_devices = models.JSONField(
        default=list,
        blank=True,
        help_text=(
            "Device fingerprints previously seen on successful logins for "
            "this account."
        ),
    )

    # Which channel(s) the customer normally uses. A session on an unexpected
    # channel is itself a weak anomaly signal worth capturing downstream.
    channel_preference = models.CharField(
        max_length=8,
        choices=CHANNEL_PREFERENCE_CHOICES,
        default=CHANNEL_BOTH,
        help_text="Channel(s) this customer normally banks through.",
    )

    # How long the account has existed, in days. Very young accounts behave
    # differently (thin history) and are treated more cautiously by the model.
    account_age_days = models.PositiveIntegerField(
        default=0,
        help_text="Age of the bank account in days at time of scoring.",
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this record was created.",
    )

    class Meta:
        verbose_name = "bank user"

    def __str__(self):
        return f"BankUser {self.user_id}"


class Session(models.Model):
    """
    One login/interaction session, on the mobile app or over USSD.

    Captures the *raw context signals* available at session start: which
    device, which SIM, which network location. These are what the feature
    engineering step turns into flags/scores on ``BehavioralFeatures``.
    """

    CHANNEL_APP = "app"
    CHANNEL_USSD = "ussd"
    CHANNEL_CHOICES = [
        (CHANNEL_APP, "Mobile app"),
        (CHANNEL_USSD, "USSD"),
    ]

    session_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique identifier for this session.",
    )

    # The protected account this session belongs to.
    user = models.ForeignKey(
        BankUser,
        on_delete=models.CASCADE,
        related_name="sessions",
        help_text="The bank account this session belongs to.",
    )

    # Channel the session came in on. USSD sessions will not have a device
    # fingerprint (feature phones), which is why that field allows NULL.
    channel = models.CharField(
        max_length=8,
        choices=CHANNEL_CHOICES,
        help_text="Channel used for this session: mobile app or USSD.",
    )

    # When the session started. Drives the hour-of-day deviation feature.
    # Uses default=timezone.now (a callable, evaluated per-instance) instead
    # of auto_now_add so the synthetic dataset generator can backdate
    # sessions to historical times by simply passing timestamp=... on create.
    timestamp = models.DateTimeField(
        default=timezone.now,
        help_text="When the session started.",
    )

    # Stable hash of device characteristics (model, OS build, etc.). Stored
    # as a hash rather than raw attributes for privacy. NULL for USSD, where
    # no rich device identity exists.
    device_fingerprint = models.CharField(
        max_length=128,
        null=True,
        blank=True,
        help_text=(
            "Hashed device fingerprint; NULL for USSD sessions on feature "
            "phones."
        ),
    )

    # Hashed SIM identity. We store a hash of the IMSI, never the raw IMSI,
    # so a database leak does not expose subscriber identities.
    sim_id = models.CharField(
        max_length=64,
        help_text="Salted hash of the SIM identity (IMSI); never stored raw.",
    )

    # Coarse network-location signal: IP address for app traffic, cell tower
    # ID for USSD. Either way it lets us detect implausible location jumps.
    ip_or_cell_tower_id = models.CharField(
        max_length=64,
        help_text="Client IP address (app) or cell tower ID (USSD).",
    )

    # City-level geohash derived from the above. Deliberately coarse: we need
    # 'same city?' answers, not surveillance-grade precision.
    location_geohash = models.CharField(
        max_length=12,
        help_text="Coarse (city-level) geohash of the session's location.",
    )

    # How long the session lasted, when it has ended. Abnormally short or
    # long sessions can indicate automation or confusion (e.g. victim being
    # coached over the phone).
    session_duration_seconds = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Session length in seconds; NULL while still open.",
    )

    # Convenience booleans stamped at ingest time, comparing this session's
    # device/SIM against the user's history (registered_devices etc.).
    is_new_device = models.BooleanField(
        default=False,
        help_text="True if this session's device was never seen before.",
    )
    is_new_sim = models.BooleanField(
        default=False,
        help_text="True if this session's SIM was never seen before.",
    )

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["user", "-timestamp"]),
        ]

    def __str__(self):
        return f"Session {self.session_id} ({self.channel})"


class Transaction(models.Model):
    """
    A money-transfer attempt nested inside a Session.

    A session may contain zero, one, or many transactions; every transaction
    always belongs to exactly one session. This nesting matters because
    velocity features (how many transfers in the last N minutes) are computed
    across sessions via their timestamps.
    """

    transaction_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique identifier for this transaction attempt.",
    )

    # The session within which this transfer was attempted.
    session = models.ForeignKey(
        Session,
        on_delete=models.CASCADE,
        related_name="transactions",
        help_text="The session this transfer attempt occurred in.",
    )

    # When the transfer was attempted. Used for velocity calculations
    # (transfers per 5-minute window) and burst detection.
    timestamp = models.DateTimeField(
        auto_now_add=True,
        help_text="When the transfer was attempted.",
    )

    # Amount requested, in Naira. Compared against the user's typical range
    # to produce the amount-deviation score.
    amount = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        help_text="Transfer amount in Naira.",
    )

    # Beneficiary identifier for the transfer destination.
    recipient_id = models.CharField(
        max_length=64,
        help_text="Identifier of the transfer beneficiary.",
    )

    # True when recipient_id is not in the user's typical_recipients list.
    # New recipients are one of the strongest single ATO indicators.
    is_new_recipient = models.BooleanField(
        default=False,
        help_text="True if this beneficiary was never paid by this user before.",
    )

    # Gap between this transfer and the user's previous one. Very small gaps
    # combined with large amounts suggest automated draining.
    time_since_last_transaction_seconds = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text=(
            "Seconds since the user's previous transaction; NULL for the "
            "first-ever transaction."
        ),
    )

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"Transaction {self.transaction_id} ({self.amount} NGN)"


class BehavioralFeatures(models.Model):
    """
    Engineered feature vector computed at scoring time, attached 1:1 to a
    Session.

    This is the exact row fed into the risk-scoring logic. It is kept as its
    own table (rather than columns on Session) so the raw signals and the
    derived features evolve independently, and so judges/reviewers can see
    precisely what the model sees.
    """

    session = models.OneToOneField(
        Session,
        on_delete=models.CASCADE,
        related_name="features",
        help_text="The session these features were computed from.",
    )

    # How far the session start hour falls outside the user's typical login
    # hours. Normalised 0..1 (0 = squarely inside normal hours).
    hour_deviation_score = models.FloatField(
        default=0.0,
        help_text="Normalised deviation of login hour from user's baseline.",
    )

    # How far the transaction amount falls outside the user's typical range,
    # normalised 0..1. NULL when the session contains no transaction.
    amount_deviation_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Normalised amount deviation; NULL if no transaction.",
    )

    # Device/SIM/location changed relative to the user's history. Note that
    # any of these alone is weak evidence (see combined flag below).
    device_change_flag = models.BooleanField(
        default=False,
        help_text="Session came from a device not seen before.",
    )
    sim_change_flag = models.BooleanField(
        default=False,
        help_text="Session used a SIM not seen before.",
    )
    location_change_flag = models.BooleanField(
        default=False,
        help_text="Session originated from a new coarse location.",
    )

    # Transfer went to a beneficiary the user has never paid before.
    new_recipient_flag = models.BooleanField(
        default=False,
        help_text="Transfer targeted a brand-new beneficiary.",
    )

    # Number of transactions by this user in the rolling 5-minute window.
    # High values indicate rapid-fire draining behaviour.
    velocity_count_5min = models.PositiveIntegerField(
        default=0,
        help_text="Transactions by this user in the last 5 minutes.",
    )

    # DELIBERATE DESIGN CHOICE:
    # This flag is True ONLY when a device-or-SIM change happens TOGETHER
    # WITH a location change — never when either occurs alone. In the Nigerian
    # context neither alone implies fraud: families legitimately share phones
    # (device change alone), and people swap SIMs routinely (SIM change alone,
    # e.g. after buying a new SIM). But a new device/SIM appearing together
    # with a new network location is the classic signature of an attacker who
    # has taken over credentials AND moved off the victim's usual footprint.
    combined_device_location_flag = models.BooleanField(
        default=False,
        help_text=(
            "True only when device-or-SIM change co-occurs with a location "
            "change; never True on either alone."
        ),
    )

    # USSD-only signal: how much menu-navigation timing deviates from the
    # user's norm (an attacker fumbling through menus is slower/hesitant).
    # NULL for app-channel sessions where there are no USSD menus.
    menu_timing_deviation_score = models.FloatField(
        null=True,
        blank=True,
        help_text="USSD menu-navigation timing deviation; NULL on app sessions.",
    )

    # When this feature vector was computed (audit trail).
    computed_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this feature vector was computed.",
    )

    def __str__(self):
        return f"Features for session {self.session_id}"

    def save(self, *args, **kwargs):
        """
        Enforce the invariant behind combined_device_location_flag at write
        time so the semantic documented above can never drift, regardless of
        which caller sets the individual flags.
        """
        credential_context_changed = (
            self.device_change_flag or self.sim_change_flag
        )
        self.combined_device_location_flag = bool(
            credential_context_changed and self.location_change_flag
        )
        super().save(*args, **kwargs)


class FraudLabel(models.Model):
    """
    Ground-truth label for a Session — SYNTHETIC DATASET ONLY.

    Exists purely to train/evaluate the detector against labelled synthetic
    attack scenarios. The real-time scoring path must NEVER read this table;
    keeping labels in a separate table makes accidental leakage structurally
    difficult and easy to audit in code review.
    """

    ATTACK_CREDENTIAL_THEFT = "credential_theft"
    ATTACK_SIM_SWAP = "sim_swap"
    ATTACK_LOW_AND_SLOW = "patient_low_and_slow"
    ATTACK_TYPE_CHOICES = [
        (ATTACK_CREDENTIAL_THEFT, "Credential theft"),
        (ATTACK_SIM_SWAP, "SIM swap"),
        (ATTACK_LOW_AND_SLOW, "Patient low-and-slow drain"),
    ]

    # The labelled session.
    session = models.OneToOneField(
        Session,
        on_delete=models.CASCADE,
        related_name="fraud_label",
        help_text="The session this ground-truth label applies to.",
    )

    # Whether this session is part of an attack scenario.
    is_attack = models.BooleanField(
        default=False,
        help_text="True if this session belongs to a simulated attack.",
    )

    # Which synthetic attack pattern the session belongs to. Only meaningful
    # when is_attack is True.
    attack_type = models.CharField(
        max_length=32,
        choices=ATTACK_TYPE_CHOICES,
        null=True,
        blank=True,
        help_text="Attack archetype for attack sessions; NULL otherwise.",
    )

    # True when the session looks suspicious but is genuinely legitimate:
    # e.g. a real customer recovering after losing their phone (real SIM
    # swap), or a family member using a shared device. These rows exist to
    # teach the model (and calibrate thresholds) that anomaly != fraud.
    is_legitimate_anomaly = models.BooleanField(
        default=False,
        help_text=(
            "Suspicious-looking but genuinely legitimate activity (shared "
            "family device, genuine post-loss SIM replacement, etc.)."
        ),
    )

    def __str__(self):
        state = "attack" if self.is_attack else "benign"
        return f"Label for session {self.session_id}: {state}"

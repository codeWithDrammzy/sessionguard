# SessionGuard — Full System Architecture & Implementation Handoff

> **Purpose of this document:** Give another AI (or a fresh developer) complete, self-contained context to understand everything we built — what the system does, which files do what, how the data flows, the exact scoring logic, and how to run/verify it. This is the definitive "read me first" for the codebase.

---

## 1. What the product is

**SessionGuard** is a real-time **behavioural Account-Takeover (ATO) detection system** for Nigerian retail banking across **two channels: mobile app and USSD (`*737#`)**. It was built for a national hackathon (~500 teams) under a challenge brief that models real Nigerian banking realities.

The core idea: instead of trusting a one-time login, every session is **scored continuously** against the customer's own learned behavioural baseline — their normal login hours, normal transfer amounts, known recipients, known devices/SIMs/locations, their typing rhythm and (on USSD) menu-navigation pace. Deviations produce a verdict: **approve / challenge (step-up verification) / block**.

**Stack:** Python 3.12, Django 5.x, Django REST Framework, scikit-learn (logistic regression), SQLite (`db.sqlite3`), a hand-rolled geohash codec, HTML/JS bank-app and "Control Room" UI. **No external ML/hardware dependencies beyond scikit-learn.**

**The brief's hard requirements we specifically honour:**
- A shared **family phone** is *not* fraud (device change alone must not hard-block).
- A **genuine SIM swap** after losing a phone is *not* fraud (real-life SIM change must not hard-block).
- New-recipient transfers are *normal life* (must not alone trigger anything harsh).
- **USSD** is a first-class, differentiating channel (attacker visibility differs: no device fingerprint).

---

## 2. High-level architecture (layers)

```
 Browser: Bank App (customer) + Control Room (analyst)
              │           │
              ▼           ▼
        /api/bank/*   /api/demo/*          <-- demo-only (presentation)
   bank_views.py   demo_views.py / demo_scenarios.py
              │
              ▼
   INGEST / 1 real ledger endpoint: /api/bank/send-money/  (BOTH channels)
              │   calls the SAME pipeline as the raw API ↓
              ▼
   /api/session-event/  (app)      /api/ussd-event/  (USSD)
        SessionEventView  ──►  run_scoring_pipeline(user, data, keystroke)
   serializers.py validates               │
                                          ▼
                              feature_engine.compute_features()
                              (persists BehavioralFeatures FIRST, so
                               keystroke/menu evidence is in THIS
                               transaction's feature vector)
                                          │
                          ┌───────────────┼────────────────┐
                          ▼               ▼                ▼
                   rules_engine     ml_model        offline_fallback
                   (weighted)      (logistic reg)   (queue; never blocks)
                          │               │                │
                          └───────────────┴────────────────┘
                                          │
                                          ▼
                              hybrid_scorer.score_session_hybrid()
                              (max(rules, ML*100) + context override)
                                          │
                                          ▼
                        verdict: approve / challenge / block
                                          │
                              explanation.py (human "why" reasons)
                                          │
                bank_send_money: approve→deduct (row-lock)
                                 challenge→HOLD for SMS verify (row-lock on release)
                                 block→never commit funds
```

**Live scoring pipeline entry point:** `core.views.run_scoring_pipeline(user, data)` — used identically by the raw API and the bank ledger endpoint, keeping the ledger consistent across channels.

---

## 3. Data model — `core/models.py`

**Design philosophy (critical to understand):** *Ground-truth labels are kept in separate tables so label leakage into the live scoring path is structurally impossible* — the production path never reads `FraudLabel` or `ConfirmedOutcome`.

| Model | Role |
|---|---|
| **`BankUser`** | The protected account + its learned behavioural baseline. UUID pk. Deliberately **not** Django's auth `User` (models a core-banking account holder, not an admin login). Fields: `typical_login_hours` (JSON list of `[start,end]` hour ranges), `typical_transfer_min/max`, `typical_recipients` (JSON), `registered_devices` (JSON), `channel_preference`, `account_age_days`. Demo-banking extras: `first_name`, `phone_number`, `balance`, and server-side PBKDF2 PIN hashes (`login_pin_hash` 6-digit, `transfer_pin_hash` 4-digit) with a one-time-setup / no-reset flow. |
| **`Session`** | One login/interaction on app or USSD — raw context: `device_fingerprint` (NULL for USSD), `sim_id` (salted hash, never raw IMSI), `ip_or_cell_tower_id` (IP for app, tower for USSD), `location_geohash`, `session_duration_seconds`. Ingest-time booleans `is_new_device` / `is_new_sim`. `timestamp` uses `default=now` (NOT `auto_now_add`) so the dataset generator can back-date history. |
| **`Transaction`** | A transfer nested inside a Session (0..many per session). `amount`, `recipient_id`, `narration`, `is_new_recipient`, `time_since_last_transaction_seconds` (drives velocity + impossible-travel timing). Demo verdict fields: `reference` (e.g. `SG-9F2C41AB`), `outcome` (approve/challenge/block). |
| **`BehavioralFeatures`** | Engineered vector, **1:1 with a Session** — the exact row the scorer reads. Stored separately from `Session` so raw signals and derived features evolve independently and judges can see exactly what the model sees. |
| **`KeystrokeDynamics`** | Per-keystroke biometrics (mean hold, mean interval, characters-per-minute) + `login_pin_failures`; NULL for USSD (no keyboard API). Historical rows form the typing baseline via a causal z-score. |
| **`FraudLabel`** | **SYNTHETIC DATASET ONLY.** Ground truth for training/eval: `is_attack`, `attack_type`, `is_legitimate_anomaly`. Never read by scoring. |
| **`ConfirmedOutcome`** | Human-confirmed outcome of a *live/demo* scored session (`confirmed_attack` with `source` demo/admin, `confirmed_by`). Folded back into the next `retrain_model`. Separate from `FraudLabel` on purpose. |
| **`RecipientDirectory`** | NUBAN account-number → display-name map for the demo "name enquiry". |

### The `combined_device_location_flag` invariant (the single most important domain decision)
Defined once in `resolve_combined_device_location_flag()` (`models.py`) and enforced both in `BehavioralFeatures.save()` **and** in the bulk feature engine (which bypasses `save()` via `bulk_create`):

> **True ONLY when a device-or-SIM change happens TOGETHER with a location change — never when either occurs alone.**

Rationale (Nigerian context): a new phone (family sharing) or a new SIM (routine upgrade) is legitimate in isolation, but new hardware/SIM **plus** a new network location is the classic takeover signature. This is the linchpin that keeps the whole "don't punish honest customers" story honest.

### Feature vector content (`BehavioralFeatures`)
- `hour_deviation_score` (0..1, how far outside typical login hours)
- `amount_deviation_score` (0..1, outside typical amount band; NULL if no txn)
- `device_change_flag`, `sim_change_flag`, `location_change_flag`
- `combined_device_location_flag` (the invariant above)
- `new_recipient_flag`
- `velocity_count_5min` (transactions in rolling 5-min window)
- `menu_timing_deviation_score` (USSD only; NULL on app)
- `keystroke_deviation_score` (app only; NULL on USSD / first few app sessions)
- `impossible_travel_flag` (implied speed from immediately-prior session > 900 km/h across a genuinely long distance)

---

## 4. Module-by-module breakdown (`core/`)

| File | Responsibility |
|---|---|
| `views.py` | `run_scoring_pipeline()` + `SessionEventView` (a single view class, channel fixed per-URL via `.as_view(channel='app'|'ussd')`). Persists session/features first, then scores. |
| `serializers.py` | DRF input validation (`SessionEventSerializer`, nested `TransactionEventSerializer`, `KeystrokeEventSerializer`). USSD drops device fingerprint; app missing-fingerprint is warn-not-fail; future `timestamp` dropped; unknown `user_id` handled in the view (→404) not the serializer (→400). |
| `feature_engine.py` | `compute_features()` — turns raw session/history into `BehavioralFeatures`. Batch-aware (works under `bulk_create`). Behavioural baselines: `MIN_KEYSTROKE_BASELINE_SESSIONS = 5` → below that, `keystroke_deviation_score` is `None`. |
| `rules_engine.py` | Deterministic weighted scoring (baseline, transparent). See §6 weights. |
| `ml_model.py` | scikit-learn `LogisticRegression` trained on 11 FEATURE_COLUMNS. Bundle caching + `reload_bundle()` for hot swap (a retrain is picked up on the next request, no restart). `predict_risk(features)` handles `None`/bool vectorisation. |
| `hybrid_scorer.py` | Final decision: rules + ML + **context-normalcy override** (see §7). |
| `offline_fallback.py` | Offline queue (`offline_queue.jsonl`, gitignored). When offline, events queue and are never hard-blocked. |
| `explanation.py` | Builds the human-readable "why" (list of triggered reasons + codes) for the Control Room. |
| `bank_views.py` | Demo customer banking layer (app + USSD simulator). One ledger endpoint, both channels. PIN/signup/login/name-enquiry/no-cache page. |
| `demo_views.py` + `demo_scenarios.py` | Demo-only Control Room: `/api/demo/scenarios/` (five one-click presets), `/api/demo/toggle-offline/`, `/api/demo/confirm-outcome/`. Presets built from REAL DB rows and pushed through the real API (no mocks). `DEMO_TOWER` marker lets demo traffic be purged. |
| `geohash_util.py` | Stdlib-only geohash encode/decode + haversine. Shared by generators and feature engine so both agree on cell boundaries. Nigerian city centres + far-international attack cities. |
| `management/commands/retrain_model.py` | Re-runs ML training folding in `ConfirmedOutcome` rows; prints eval; saves bundle; optional `--no-reload`. |
| `management/commands/reset_demo.py` | Demo-hygiene: deletes browser test accounts, demo/smoke debris, same-day unlabelled sessions, clears offline queue. Never touches `FraudLabel` rows. `--check` pre-views. |

**Dataset generator (separate, committed):** `dataset_generator/`
- `generate_users.py` — 250 `BankUser` baselines. Seed 42.
- `generate_sessions.py` — normal (non-fraud) session/transaction history per user. **Seed 43** (independent stream).
- `inject_attacks.py` — attack sessions labelled via `FraudLabel`. **Seed 44.** Attack archetypes: `credential_theft` (loud), `patient_low_and_slow` (quiet device-only), `sim_swap_takeover` (USSD-native).
- `inject_legitimate_anomalies.py` — `family_shared_phone` (6) + `genuine_sim_swap` (4) **Seed 45** (four fully independent RNG streams so no dataset can silently correlate with another).

**DB snapshot:** committed `db.sqlite3` = **250 customers / 2237 sessions / 1716 transactions** (matches `reset_demo` output and DEMO_SCRIPT.md). `core/trained_model.joblib` is the committed trained bundle (42 positive examples).

---

## 5. URLs (all routes live in `sessionguard_project/urls.py` — there is NO `core/urls.py`)

```
admin/                                Django admin
api/session-event/                    SessionEventView(channel='app')   <-- real scoring
api/ussd-event/                       SessionEventView(channel='ussd')  <-- real scoring
api/demo/scenarios/                   demo_scenarios         (Control Room presets)
api/demo/toggle-offline/              toggle_offline
api/demo/confirm-outcome/             confirm_outcome
demo/                                 ControlRoomView        (analyst UI)
bank/                                 BankAppView            (customer app + USSD sim)
api/bank/signup/  /login/  /set-pin/  /verify-pin/  /state/<uuid>/
api/bank/send-money/  /lookup-account/
```

---

## 6. Rules engine — `rules_engine.py` (the transparent baseline)

**Score bands:** `0–29 → approve`, `30–59 → challenge` (step-up/OTP), `≥60 → block`. Weights are **hand-tuned from first principles** (how damning each signal is independently) deliberately before looking at eval data — an honest baseline, not optimised.

```
WEIGHTS = {
  "combined_device_location": 45,   # context AND location moved together
  "impossible_travel":        50,   # physically impossible speed from prior session
  "device_change_alone":      10,   # weak alone (family phone)
  "sim_change_alone":         10,   # weak alone (routine SIM upgrade)
  "location_change_alone":     5,
  "hour_deviation_max":       15,   # scaled: score * max
  "amount_deviation_max":     20,
  "menu_timing_deviation_max":15,   # USSD pacing
  "keystroke_deviation_max":  12,
  "velocity_per_extra_session": 8,  # per txn BEYOND the first in 5-min window
  "new_recipient_alone":       3,   # deliberately tiny: normal life
}
```

**Key design notes:**
- `combined_device_location` **subsumes** its parts — no double counting when it fires.
- `new_recipient_alone` is deliberately just +3: alone it never moves a verdict.
- Velocity counts only sessions **beyond the first** in the window (`max(0, count-1) * 8`) so a single transfer is not punished.
- Scaled rules (hour/amount/menu-timing) contribute `round(score * max_points)`.

---

## 7. Hybrid scorer — `hybrid_scorer.py` (the production decision)

**`score_session_hybrid(features)`** does, in order:
1. **Rules score** via `score_session()`.
2. **ML probability** via `predict_risk()`; convert to points `round(ml_prob * 100)`.
3. **Combined = `max(rules_score, ml_points)`** — NOT an average. Averaging would dilute a strong true-positive signal from either engine; each engine's certainty should be able to carry the decision alone.
4. **Band** into approve/challenge/block.
5. **THE CONTEXT-NORMALCY OVERRIDE** (the crux):
   ```
   if verdict == "block"
      AND (device_change_flag OR sim_change_flag)
      AND is_context_normal(features):
        verdict = "challenge"        # cap block → challenge ONLY
   ```
   `is_context_normal()` is true when the session's behaviour besides the hardware change is ordinary: `amount_deviation_score ≤ 0.15`, `hour_deviation_score ≤ 0.15` (with `impossible_travel_flag` forced to veto, and new-recipient deliberately **excluded** from normalcy).

**Why the override exists:** the ML model, trained on this dataset, learned "device/SIM changed → fraud" so aggressively (coeffs ≈ +6.6/+5.9) that genuine SIM-swap recoveries scored p≥0.94 and would have been blocked 4/4. The rules engine approved those but was blind to patient attacks. Insight: **hardware change is ambiguous on its own** — its meaning depends on whether everything else looks ordinary.
- *Patient attacker* → changes hardware, careful elsewhere → deserves interception (gets challenge/friction).
- *Genuine recovery* → changes hardware, everything else IS their normal life → must never hard-block (gets challenge, an OTP, not a frozen account).

Only "block" is softened, by exactly one level, **never** the reverse, and never when impossible-travel is present (that is the opposite of ordinary).

---

## 8. ML model — `ml_model.py`

- **Model:** interpretable `LogisticRegression` (choice justified: judges/reviewers can read the coefficients and defend the logic — no black box).
- **Features (11 FEATURE_COLUMNS):** the `BehavioralFeatures` vector (booleans + scaled scores; `None` → 0.0, `bool` → int — handled inside `predict_risk`).
- **Trained on 80%** of the labelled dataset; the committed bundle has **42 positive (attack) training rows**.
- **Committed coefficients** (strongest → weakest): `sim_change +6.48`, `device_change +4.84`, `keystroke +3.33`, `new_recipient +1.79`, `menu_timing +0.96`, `impossible_travel +0.60`, `location -0.41`, `amount -0.20`, `combined +0.20`, `hour -0.07`, `velocity -0.03`.
- **Continuous learning:** `ConfirmedOutcome` rows from the Control Room (judge/analyst confirms "was this really an attack?") are folded in on the next `python manage.py retrain_model`, and `reload_bundle()` makes a running server adopt the new model on its next request.

---

## 9. Bank ledger & holds — `bank_views.py` (how money actually moves)

`/api/bank/send-money/` is the **single** ledger endpoint serving BOTH channels:
1. Validates the body, builds the scoring `data`, calls `run_scoring_pipeline(user, data)` (the same pipeline as the raw API).
2. **Funds-availability check happens BEFORE scoring** (an accounting question, not a fraud question) — amount > balance short-circuits to block.
3. On verdict:
   - **approve** → set `reference`, mark outcome approve, **deduct under a row lock** (`select_for_update`) and re-validate balance inside the lock (closes the read-then-deduct race between concurrent requests; if funds ran out, downgrade to block).
   - **challenge** → mark outcome challenge, set `reference`, and place a **CHALLENGE_HOLDS** entry (in-memory `OrderedDict`, capped at 200). The transfer is **not committed**.
   - **block** → mark outcome block; funds never move.
4. **Hold-release path:** the customer completes SMS-style verification (demo: any numeric code) by resubmitting with `challenge_reference`; the hold is validated, the balance re-checked under a row lock, and only then the transfer commits (approve) — mirroring how step-up verification releases a held payment at real banks.
   - **Design honesty:** holds are in-memory (`OrderedDict`), so a server restart drops unverified holds exactly like an unclaimed OTP expiring. Bounded at `HOLD_LIMIT=200`.

**PIN handling (server-side, decided):** login PIN (6-digit) and transfer PIN (4-digit) are PBKDF2-hashed on the server via `make_password()` and stored on `BankUser`; the browser only collects digits and posts them; correctness is decided with `check_password()`. One-time setup (`bank_set_pin`), **no forgot/reset/recovery flow by explicit design** (a half-setup pair can be re-run to finish; a fully-set pair refuses to overwrite). A null hash answers `pin_not_set` so the app routes through one-time setup.

---

## 10. The five demo archetypes (`demo_scenarios.py`)

Each Control Room preset is a **complete payload matching the serializer** built from **real DB rows**, pushed through the **real** `/api/session-event/` or `/api/ussd-event/` (no mocks). Distinct users per preset so rapid-fire demoing never contaminates behavioural baselines.

| Preset | Channel | Narrative | Expected |
|---|---|---|---|
| `normal_login` | app | own phone/SIM/location/usual recipient, typical amount | **approve** |
| `obvious_attack` (SIM-swap takeover) | **ussd** | new tower, new location, 5× largest transfer, brand-new beneficiary | **block** |
| `patient_attack` | app | ONLY device changed; SIM/location/hour/amount normal (replayed in-window) | **challenge** (friction) |
| `family_sharing` | app | same phone/SIM/location, small amount to a new person | **approve** (≤1 gentle verify) |
| `genuine_simswap` | app | REAL revenue: new device AND new SIM, but at home, in hours, known beneficiary (replayed in-window) | **challenge** (never block) |

USSD choice is deliberate: the brief positions USSD as the differentiating Nigerian channel, and SIM-swap takeover is its signature attack — showing a block on the USSD path (rendered as a green-screen terminal) demonstrates both in one click. Preset events carry the inert `DEMO_TOWER` marker so demo traffic can be wiped safely.

---

## 11. Verified evaluation numbers (live, from committed code)

| Scorer | Precision | Recall | FPR | patient strict | patient op. | simswap FP |
|---|---|---|---|---|---|---|
| **rules-only** | 100% | 59.5% | 0% | — | — | 0/4 |
| **hybrid** | 100% | 59.5% | 0% | 12/12 | 12/12 | **0/4** |
| **ML-only @0.5** | 100% | 87.5% | 0% | — | — | (would block recoveries — hence the override) |

- **Legitimate anomalies:** `family_shared_phone` → zero hard-blocks; `genuine_sim_swap` → zero hard-blocks (both protected by the invariant + override).
- **Operational catch rate** counts challenge+block together (a challenge still stops the transfer until verification passes).
- **Eval caveat (stated honestly in-code):** the hybrid/ML report runs over the full dataset with the ML trained on 80% of it → illustrative comparison vs the rules baseline, not a clean generalisation metric.

**Diagnostic finding (expected, not a bug):** fast typing alone is a weak signal. Below `MIN_KEYSTROKE_BASELINE_SESSIONS=5` priors, `keystroke_deviation_score=None`; even at max, `keystroke_deviation_max=12 <` the challenge threshold (30), so keystrokes alone never decide a verdict — they only add weight in combination. Also the `wlPhone` login field is **not** tracked by the JS keystroke recorder (a genuine, known coverage gap for login-screen typing).

---

## 12. How to run / verify

```bash
# Django checks + the 10-test behavioural-guarantee suite
python manage.py check
python manage.py test core            # 10 tests, ~0.1s, in-memory DB
python smoke_test_api.py              # end-to-end pipeline smoke test

# Dataset generators (reproducible, independent seeds 42/43/44/45)
python dataset_generator/generate_users.py
python dataset_generator/generate_sessions.py
python dataset_generator/inject_attacks.py
python dataset_generator/inject_legitimate_anomalies.py
python dataset_generator/inject_legitimate_anomalies.py   # (anomalies)

# Evaluation reports (self-contained prints)
python core/rules_engine.py
python core/hybrid_scorer.py

# Operations
python manage.py retrain_model        # fold confirmed outcomes into the ML model
python manage.py reset_demo           # restore the clean 250-customer snapshot
python manage.py reset_demo --check   # preview what reset would remove

# Local dev server -> http://127.0.0.1:8000/bank/ and /demo/
python manage.py runserver
```

---

## 13. Testing philosophy (`core/tests.py`) — the "behavioural guarantees"

The 10-test suite codifies the non-negotiables (what the demo/build would be dishonest without):
- Feature **causality** (changing a raw signal changes the right feature).
- Keystroke **baseline gate** (fewer than 5 priors → score is None).
- Hybrid **context override** (hardware change + normal context → challenge, *not* block; attack + abnormal context → block).
- Offline **never blocks**.
- ML **vectorisation** (None→0.0, bool→int, FEATURE_COLUMNS match the bundle) and rules weight logic.

---

## 14. Git / repo state

- Branch `main`, remote `https://github.com/codeWithDrammzy/sessionguard.git`.
- HEAD commits: `31ef14e` "Restore seeded snapshot (250 customers / 2237 sessions / 1716 txns)" and `4467e08` "Add behavioural-guarantee test suite" — both pushed. Working tree clean.
- Tracked: `db.sqlite3` (committed clean snapshot) + `core/trained_model.joblib` (committed). Gitignored/untracked: `offline_queue.jsonl`, `__pycache__`/`.pyc`.
- Note: LF→CRLF warnings and a benign PowerShell "RemoteException" on `git push` are cosmetic — pushes succeed.

---
*End of handoff. This document is generated from the committed code and reflects the verified, live behaviour of the system.*

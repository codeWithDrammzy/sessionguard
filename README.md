# SessionGuard

**Demonstrates a behavioural risk detection pipeline for identifying suspicious account activity and applying graduated responses — mobile app and USSD banking.**

Built for the ICSC 2026 Universities Hackathon — Track A: Financial Services & Digital Payments — Challenge: *Spotting Account Takeover from Behaviour*. Team: **BlueTeam**.

---

## The Problem

Most banking security checks one thing: did you enter the correct PIN or code? Once a PIN is phished, a SIM is swapped, or a criminal logs in from a new device, that check has nothing left to catch them with. By the time a fraudulent transfer clears, the money is gone.

SessionGuard adds a second question to every login and transfer: **does this still look like this specific customer**, based on their own history — not a generic profile?

## How It Works

Every customer has a behavioural profile built from their own past activity: typical login hours, typical transfer amounts, regular recipients, registered devices/SIM, and typing rhythm. Every new session is scored against **that customer's own baseline**, using three layers:

1. **Rules engine** (`core/rules_engine.py`) — a transparent, hand-written checklist. Each warning sign adds fixed points; the total is banded into `APPROVE` (0–29), `CHALLENGE` (30–59), or `BLOCK` (60+).
2. **Machine learning model** (`core/ml_model.py`) — a Logistic Regression classifier trained on 11 behavioural features, chosen deliberately for interpretability. It independently learned that a first-time SIM is the single strongest predictor of takeover, with a first-time device close behind — exactly the signals that catch a "patient" attacker who changes only their device while keeping everything else normal.
3. **Hybrid scorer** (`core/hybrid_scorer.py`) — the final score is the higher of the two layers. A **fairness override** then checks: has only the hardware changed, while amount, time, and location remain normal? If so, a `BLOCK` is automatically softened to `CHALLENGE` — a genuine customer (e.g. one who lost their phone) is never locked out, while an actual attacker is still stopped from moving money without confirmation.

Full architecture detail is in [`SessionGuard_Architecture.docx`](./SessionGuard_Architecture.docx) (or the equivalent doc in this repo) and the technical write-up submitted alongside this project.

## Key Results

Measured with a **strict, de-leaked evaluation**: 5×10 repeated stratified cross-validation (50 folds) where every held-out session's features are recomputed using only training-fold history — no leaks, no in-sample optimism. Full table and summed confusion matrices are in `ARCHITECTURE.md §11a`.

| Scorer | Precision | Recall | F1 | PR-AUC | FPR* |
|---|---|---|---|---|---|
| Rules only (CV) | 100.0% ± 0.0 | 57.6% ± 14.9 | 0.719 | **0.827 ± 0.090** | 0.000% |
| ML only (CV) | 60.3% ± 9.7 | 98.8% ± 3.6 | 0.744 | **0.923 ± 0.054** | 1.29% |
| Hybrid (CV) | 99.3% ± 3.6 | 51.9% ± 16.5 | 0.666 | **0.922 ± 0.057** | 0.009% |

\*False positive rate = benign sessions hard-blocked / all benign sessions, pooled over the same 50 folds: 0 / 22430, 289 / 22430, and 2 / 22430 for rules, ML, and hybrid respectively. A `challenge` stops a transfer until step-up verification but is **not** counted as a false positive here.

**Read this honestly:** PR-AUC (average precision) is the primary metric at a 1.84% positive rate. The high-recall ML model over-blocks (it hard-blocks most genuine SIM-swap recoveries 40/40); the hybrid's fairness override fixes that — restoring precision to 99.3% and cutting SIM-swap false-blocks to **0/40** — while keeping the *operational catch rate* (block **or** challenge, both of which stop a transfer until step-up verification) at **99.3%** of attacks. Rules running **alone** never hard-block any of the 10 legitimate anomalies (0 false blocks in 100 pooled anomaly folds); the hybrid retains 2 residual family false-blocks over 50 folds (documented limitation — the override legally cannot fire when no device/SIM changed).

**Reproducing these numbers:** they are measured on the committed 250-customer / 2285-session baseline snapshot. Run `python manage.py reset_demo` **before** re-running the headline metrics to restore that baseline — any demo or smoke-test traffic added to the DB shifts the figures (with just 3 extra unlabelled sessions, rules precision drops to ~96.6% and hybrid to ~95.1%). Then run: `python core/eval_cv.py --splits 5 --repeats 10`.

## Real-World Conditions Handled

- **Power/network cuts** — falls back to a local, cached-profile check capped at `CHALLENGE` (never a hard block on an unverifiable guess), then queues the event for full re-scoring once connectivity returns.
- **USSD / feature phones** — the full pipeline runs on `*737#` sessions too, where no device fingerprint exists; SIM identity and location carry more weight there instead.
- **Non-technical customers** — every `CHALLENGE`/`BLOCK` comes with a plain-English, two-reason explanation, never a silent block.
- **Shoulder-surfing** — the on-screen PIN keypad shuffles its 1–9 digits on every render.
- **Continuous improvement** — a human-confirmed feedback loop (`ConfirmedOutcome` model + `python manage.py retrain_model`) folds real verdicts back into training.

## Project Structure

```
sessionguard/
├── core/
│   ├── feature_engine.py       # Computes behavioural signals from raw session data
│   ├── rules_engine.py         # Hand-written scoring checklist
│   ├── ml_model.py             # LogisticRegression training + inference
│   ├── hybrid_scorer.py        # Combines rules + ML, applies fairness override
│   ├── explanation.py          # Plain-language customer explanations
│   ├── offline_fallback.py     # Degraded-mode scoring + resync queue
│   ├── models.py               # Data model (BankUser, Session, Transaction, etc.)
│   ├── views.py / bank_views.py / demo_views.py   # API endpoints
│   ├── management/commands/    # reset_demo, retrain_model
│   └── templates/
│       ├── bank/bank_app.html      # Mobile app + USSD simulator
│       └── demo/control_room.html  # Judge/analyst dashboard
├── dataset_generator/          # Deterministic synthetic data pipeline (seeded)
├── smoke_test_api.py           # Manual regression check
└── DEMO_SCRIPT.md              # Live demo walkthrough
```

## Data

All data is **synthetic**, generated by our own deterministic, seeded pipeline — no real customer data was used anywhere, per competition rules. Four independently-seeded stages:

1. `generate_users.py` (seed 42) — 250 fictional customer profiles
2. `generate_sessions.py` (seed 43) — ~3 weeks of baseline session history per customer
3. `inject_attacks.py` (seed 44) — 42 attacks across 3 archetypes (credential theft, patient low-and-slow, SIM-swap takeover)
4. `inject_legitimate_anomalies.py` (seed 45) — 10 legitimate edge cases (family shared phone, genuine SIM-swap recovery)

Fully reproducible — running the same seeds regenerates an identical dataset.

## Running It Locally

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

python manage.py migrate
python manage.py reset_demo      # resets to a clean 250-customer demo snapshot
python manage.py runserver
```

Then visit:
- `http://127.0.0.1:8000/bank/` — the mobile app / USSD simulator
- `http://127.0.0.1:8000/demo/` — the Control Room judge dashboard

Tests: `python manage.py test core` — a 30-test behavioural-guarantee suite (feature causality, fairness override, offline logic, USSD transfer-PIN guard, read-only balance checks, preset stability).

Built entirely with free, open-source tools — Python, Django, Django REST Framework, scikit-learn, SQLite. No paid services or infrastructure required.

## Known Limitations (Stated Honestly)

- Only 42 labelled attacks exist in the training dataset — enough to demonstrate the approach, not yet a statistically rigorous sample.
- The step-up confirmation step is simulated (accepts any well-formed code) to demonstrate the flow; a production system would connect a real SMS/USSD OTP provider, ideally on a channel independent of the primary device.
- New customers are scored "blind" on typing rhythm until roughly 5 prior sessions exist.
- The context-normalcy override can only fire when a device/SIM actually changed; 2 residual family false-blocks remain in hybrid across 50 CV folds.
- The `wlPhone` login field on the demo app is not tracked by the JS keystroke recorder (known coverage gap for login-screen typing).

See the full technical write-up for the complete limitations discussion and what we'd do next for each.

## Team

**BlueTeam** — ICSC 2026 Universities Hackathon, Track A.

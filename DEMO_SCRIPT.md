# SessionGuard — Live Demo Script (Stage Plan)

Everything below runs against the **real** scoring API with **real** rows in the
seeded database. No mocks, no canned responses. Tune the runtime: the whole
script is ~5 minutes; each Control Room click is one API call.

Before the demo, from the project root:

    python manage.py reset_demo     # -> exactly 250 customers, 2237 sessions
    python manage.py runserver      # start the server

Need a fallback if runserver is already live with stale code? Stop it, run
`python manage.py runserver` again, retest one scenario.

---

## 0. Cold open (30 sec) — the hook, before opening the Control Room

> "We built SessionGuard to catch account takeover on **every** channel a
> Nigerian bank customer uses — the mobile app and legacy USSD *737#* — and to
> stay honest about the hard cases, not just the obvious ones."

Show the bank app (`/bank/`):
- Note the **anti-shoulder-surf keypad**: the digit positions re-shuffle every
  time the pad renders, so someone watching a phone over a shoulder in the bus
  cannot reconstruct the PIN from hand position. **This is free real security
  for a low-resource, public-transport reality.**

Open the **Control Room** (`/demo/`). Flip the mock-up: never cache, always
live.

---

## 1. The obvious case — an attack it must catch (45 sec)
Click **"SIM-swap takeover (USSD)"** (obvious_attack).

Expected:
- Verdict **block**, score **98**, ML probability **0.98** (reserve these numbers).
- Judge view: a *new SIM*, *new tower*, *new location*, *5× the user's largest
  transfer*, to an *unknown recipient*.

Spoken point:
- "Cloned SIM on *737#*, 5× their biggest transfer, brand-new beneficiary —
  every signal screaming. Straight to **block**, no friction for the real owner."

---

## 2. The hard case — the one that beats naive rules (60 sec)
Click **"Patient attacker"** (patient_attack).

Expected:
- Verdict **challenge** (not block), score **46**, ML **0.46**.

Spoken point (the honest, high-value story):
- "Only ONE thing is odd: the **device** changed. SIM, location, hour, amount
  are all boringly normal. A simple rules engine scores this 0/12 — it would
  wave the attacker through. Our learned model sees the lone device change and
  draws **friction, not a lockout**."
- "Why challenge and not block? Because this shape also describes a genuine
  customer on a new phone. We **soften** it rather than freeze the account."
- Then confirm: click **"✓ Confirmed fraud"** on this event.

---

## 3. The genuine customer who looks guilty (45 sec)
Click **"Genuine SIM swap"** (genuine_simswap).

Expected:
- Verdict **challenge**, score **100** (block-worthy!), ML **1.0**.
- Judge view: a new device AND new SIM — but at home, in their hours, paying a
  known beneficiary.

Spoken point (the override, the standout mechanism):
- "Raw score is 100 — a normal model would **hard-block** this. But it's a real
  customer who just recovered their lost phone. The context-normalcy override
  caps it to **challenge**, so she proves it's her instead of being frozen out
  of her own money."
- Confirm **"✓ Confirmed genuine"** on this event.

---

## 4. The life event that must NOT be blocked (45 sec)
Click **"Family shared phone"** (family_sharing).

Expected:
- Verdict **approve**, score **22**, ML **0.03**.

Spoken point:
- "Mum sending money via her daughter's phone. Same SIM, same location, small
  amount, one new recipient. Approved — a customer segment the brief called
  out as routinely mis-flagged."

---

## 5. The normal baseline (30 sec)
Click **"Normal login"** (normal_login).

Expected:
- Verdict **approve**, score **2**, ML **0.01**.

Spoken point:
- "And the great majority of traffic — an ordinary login, ordinary transfer —
  sails through with a score of 2. The system is strict where it must be,
  invisible where it should be."

---

## 6. The real-world constraint: offline / low-connectivity (60 sec)
Flip the **Network Status: OFFLINE** switch (this is the `toggle-offline` API,
the demo's simulated outage).

- Run any scenario again (e.g. obvious_attack): the system returns the
  **degraded** flag — scored from a **local cached profile** using the rules
  engine only, with the event **queued for resync**.
- The verdict is capped at **challenge** at worst — never a hard block from an
  unverifiable local decision.
- Flip back ONLINE. Note the reply: full hybrid scoring resumes.
- Honest framing: *"The full ML is too heavy to run on a feature phone's
  network — so we degrade gracefully and re-score later against the live model,
  comparing degraded vs full verdicts. We chose conservatism over silence."*

---

## 7. Close the loop — "the model keeps learning" (60 sec)
This is the **new** deliverable.

- Earlier you clicked two "Confirmed …" buttons. The Judge view shows a running
  tally (e.g. "confirmed: 2 (1 fraud)"). That is **human-confirmed ground
  truth**, written to `ConfirmedOutcome`.
- Open a terminal and run:

      python manage.py retrain_model

- The report prints: `Confirmed-outcome folds: N (x attack / y benign) from
  live demo reviews`, followed by the fresh test metrics and coefficients.

Spoken point:
- "Fraud detection isn't a one-shot model. Every challenged event a reviewer
  confirms becomes **real** training data, folded back on retrain. The model
  literally sharpens as the system is used — learning **only** from confirmed
  outcomes, not from 'whatever the bank approved', so it can't drift into
  rubber-stamping everything."

Then **Reset for the next shopping cart** if you expect the judges to poke
around: `python manage.py reset_demo`.

---

## Honest-limitations cards (have these ready, unprompted if a judge asks)

1. **Small labeled set.** Only 42 synthetic attack rows; ML test metrics are
   single-attack steps. That's why rules stay the primary detector and ML is a
   targeted catch for the patient cases. (See `ml_model.py` docstring.)
2. **Hybrid eval is not a strict held-out test.** The three-way comparison is
   labelled "illustrative". The rules baseline is genuinely held-out; the ML
   component's held-out numbers are in its threshold sweep.
3. **Keystroke rhythm needs history** — brand-new app users are scored blind on
   that signal until ~5 prior keystroke sessions exist (a documented default).
4. **The demo step-up is simulated** — the challenge holds and the OTP box
   exist; a real deployment wires it to an actual OTP/SMS gateway.
5. **Data is synthetic.** Real traffic would bring real noise; the pipeline and
   the retrain hook are built so a real deployment can swap in confirmed labels
   over time.

---

## One-line pitch (for the rubric's "problem + approach" line)

> "SessionGuard: real-time account-takeover detection across app and USSD,
> with interpretable explanations, a secure anti-shoulder-surf keypad, graceful
> offline degradation, and a model that keeps learning from confirmed outcomes."

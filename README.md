# Dyania Health Hackathon 2026 — Team Submission Repository

**Challenge:** Build a study — using machine learning, a statistical model, or whatever approach you prefer — proposing a protocol to predict aortic valve durability in patients with a bioprosthetic aortic valve replacement.
**Event:** September 15–17, 2026 (3 days)
**Team size:** 2–3 ML engineers
**Team:** `[Your team name]`
**Members:** `[Name — Role]`, `[Name — Role]`, `[Name — Role]`

---

## Problem Statement

> *Fill in: 2–3 sentences framing the clinical problem your team is solving. What is the durability/failure detection gap for aortic valve replacement patients? Who is affected? What is the cost of late identification of structural valve deterioration?*

## Our Approach

> *Fill in: Briefly describe your team's strategy. What data sources are you targeting (echocardiography, implant registries, follow-up visits, imaging)? What ML approach did you choose and why? What makes your risk-prediction design clinically actionable for surveillance scheduling?*

## Key Design Decisions

> *Fill in: List 3–5 deliberate choices your team made (e.g., model choice, feature selection philosophy, how you defined structural valve deterioration as ground truth, how you handled censored/time-to-event data, how you handled missing echo follow-ups). For each, explain the reasoning.*

| Decision | Rationale |
|---|---|
| | |
| | |
| | |

---

## Getting Started

### 1. Fork this repository

Go to **[https://github.com/dyaniahealth/dyania-hackathon-avr-durability](https://github.com/dyaniahealth/dyania-hackathon-avr-durability)** and click **Fork** (top-right) to create a copy under your own GitHub account.

### 2. Clone your fork

```bash
git clone https://github.com/<your-username>/dyania-hackathon-avr-durability.git
cd dyania-hackathon-avr-durability
```

### 3. Create your team branch

Branch names must follow this format: `team/<your-team-name>` (lowercase, hyphens for spaces).

```bash
git checkout -b team/your-team-name
```

Examples: `team/panathinea`, `team/valve-guardians`, `team/svd-sentinels`

### 4. Work on your branch

Edit the template files inside `protocol/`, `model/`, `data/`, and `presentation/`. Every `> *Fill in:*` block is a placeholder — replace it with your team's content.

```bash
# Stage and commit as you go
git add .
git commit -m "your message"
```

### 5. Submit — open a Pull Request before the deadline

Push your branch to your fork and open a Pull Request to the original repository:

```bash
git push origin team/your-team-name
```

Then go to your fork on GitHub and click **"Compare & pull request"**.
Set the base repository to `dyania-health/dyania-hackathon-avr-durability` and the base branch to `main`.
Title your PR: `Team submission: <your-team-name>`

> **Deadline: September 17, 2026 — before the presentation session.**
> Only the last commit pushed before the deadline will be evaluated.
> Make sure your PR is open — **do not** merge it.

---

## Repository Structure

```
.
├── README.md                        # This file — team overview and key decisions
├── protocol/
│   └── study_protocol.md            # Full study design (main deliverable)
├── ml/
│   └── approach.md                  # ML methodology and validation strategy
├── data/
│   └── data_plan.md                 # Data sources, preprocessing, availability
├── presentation/
│   └── slides.pdf                   # 5–10 slide deck for expert panel
└── notebooks/                       # (Optional) Proof-of-concept implementation
```

---

## Submission Checklist

- [ ] `README.md` — team overview, problem framing, key design decisions
- [ ] `protocol/study_protocol.md` — complete study protocol
- [ ] `ml/approach.md` — ML methodology
- [ ] `data/data_plan.md` — data plan
- [ ] `presentation/slides.pdf` — slide deck
- [ ] `notebooks/` — proof-of-concept (optional, evaluated positively if present)

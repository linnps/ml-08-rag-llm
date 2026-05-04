"""
Self-authored synthetic corpus for a RAG demo.

The corpus is the invented internal handbook of a fictional company —
"Aetherline Logistics" — with entries on policies, products, people,
and procedures. Every fact is something we wrote ourselves, so:
- there are no copyright concerns,
- we know the ground truth for every test question, and
- we can build adversarial out-of-corpus questions (questions whose
  answer is *not* in the corpus) to measure how often the system
  hallucinates instead of refusing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# ----------------------------------------------------------------- corpus --
# Each "document" is a short article. The chunker (in train.py) splits
# them into ~1-3 sentence chunks before embedding.

DOCUMENTS = [
    {
        "doc_id": "policy-vacation",
        "title": "Vacation policy",
        "text": (
            "Aetherline Logistics offers all full-time staff 22 paid vacation days per year. "
            "Vacation days accrue monthly and unused days roll over to the next calendar year, "
            "up to a maximum of 10 carried-over days. Vacation requests must be submitted at "
            "least two weeks in advance through the HR portal. Managers approve requests within "
            "three business days."
        ),
    },
    {
        "doc_id": "policy-remote",
        "title": "Remote work policy",
        "text": (
            "Aetherline supports hybrid work for all knowledge-worker roles. Staff are expected "
            "to be in the office a minimum of two days per week, typically Tuesday and Thursday. "
            "Fully remote arrangements are available for staff living more than 80 km from the "
            "nearest office, subject to manager approval and an annual review."
        ),
    },
    {
        "doc_id": "product-relayframe",
        "title": "RelayFrame fleet management product",
        "text": (
            "RelayFrame is Aetherline's flagship fleet-management product. It tracks vehicle "
            "telemetry in real time, schedules predictive maintenance, and integrates with most "
            "major ERP systems via REST APIs. The current version, 4.3, was released in March 2026. "
            "RelayFrame supports up to 10,000 vehicles per tenant in its Enterprise tier."
        ),
    },
    {
        "doc_id": "product-cargolens",
        "title": "CargoLens visibility product",
        "text": (
            "CargoLens is Aetherline's shipment visibility product, sold to retailers and 3PLs. "
            "It uses GPS, BLE beacons, and IoT sensors to provide door-to-door tracking with "
            "an SLA of 95% on-time arrival prediction accuracy. Pricing starts at 0.18 USD per "
            "tracked shipment for volumes under one million per month."
        ),
    },
    {
        "doc_id": "people-leadership",
        "title": "Leadership team",
        "text": (
            "Aetherline's CEO is Mira Tan, who founded the company in 2017. The CTO is Jordan "
            "Reyes, who previously led the routing platform at NorthArc Freight. The VP of "
            "People is Anh Nguyen. The board chair is Dr. Lukas Vetter, formerly of the Hamburg "
            "Logistics Institute."
        ),
    },
    {
        "doc_id": "procedure-incident",
        "title": "Incident response procedure",
        "text": (
            "Production incidents at Aetherline are graded P0 through P3. P0 incidents (full "
            "outage of a paid service) require an on-call engineer to acknowledge within "
            "10 minutes and a postmortem to be drafted within 5 business days. P1 incidents "
            "(major degradation) follow the same on-call SLA but allow 10 business days for the "
            "postmortem. The incident commander is always the senior on-call engineer, never the "
            "manager on-call."
        ),
    },
    {
        "doc_id": "procedure-onboarding",
        "title": "New-hire onboarding procedure",
        "text": (
            "Every new hire at Aetherline completes a two-week onboarding rotation. Week one is "
            "spent in cross-functional shadowing across Sales, Support, Engineering, and Operations. "
            "Week two is role-specific. All new hires receive a laptop and a phone, plus access to "
            "the internal wiki, the issue tracker, and the deployment dashboard on day one."
        ),
    },
    {
        "doc_id": "policy-expense",
        "title": "Expense policy",
        "text": (
            "Aetherline reimburses business expenses up to 250 USD without receipts and any "
            "amount with receipts, submitted within 30 days. Travel for client meetings is "
            "always reimbursable. Coach airfare is the default; business class requires VP "
            "approval and is allowed on flights longer than 8 hours. Per-diem meal allowances "
            "are 60 USD domestic and 90 USD international."
        ),
    },
]


# Test questions paired with the answer-bearing doc_id.
# Adversarial: some questions intentionally have no answer in the corpus
# (out_of_corpus=True) to test refusal behaviour.
QUESTIONS = [
    {"q": "How many vacation days do full-time staff get per year?",
     "doc_id": "policy-vacation", "out_of_corpus": False},
    {"q": "How far in advance must vacation requests be submitted?",
     "doc_id": "policy-vacation", "out_of_corpus": False},
    {"q": "What days of the week are people expected in the office?",
     "doc_id": "policy-remote", "out_of_corpus": False},
    {"q": "How many vehicles per tenant can RelayFrame Enterprise tier support?",
     "doc_id": "product-relayframe", "out_of_corpus": False},
    {"q": "What is the SLA target for CargoLens on-time arrival prediction?",
     "doc_id": "product-cargolens", "out_of_corpus": False},
    {"q": "Who is the CTO of Aetherline?",
     "doc_id": "people-leadership", "out_of_corpus": False},
    {"q": "When did Mira Tan found the company?",
     "doc_id": "people-leadership", "out_of_corpus": False},
    {"q": "What is the on-call acknowledgement SLA for a P0 incident?",
     "doc_id": "procedure-incident", "out_of_corpus": False},
    {"q": "How long is the new-hire onboarding rotation?",
     "doc_id": "procedure-onboarding", "out_of_corpus": False},
    {"q": "What is the default airfare class for business travel?",
     "doc_id": "policy-expense", "out_of_corpus": False},
    # Out-of-corpus questions — refusal is the correct answer.
    {"q": "What is Aetherline's stock ticker?",
     "doc_id": None, "out_of_corpus": True},
    {"q": "How many employees does Aetherline have in 2026?",
     "doc_id": None, "out_of_corpus": True},
    {"q": "Who is Aetherline's chief financial officer?",
     "doc_id": None, "out_of_corpus": True},
    {"q": "What is the discount for educational customers of CargoLens?",
     "doc_id": None, "out_of_corpus": True},
]


def main() -> None:
    p = argparse.ArgumentParser(description="Save the synthetic Aetherline corpus.")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "corpus.jsonl", "w") as f:
        for d in DOCUMENTS:
            f.write(json.dumps(d) + "\n")
    with open(args.out_dir / "questions.jsonl", "w") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q) + "\n")
    print(f"Saved {len(DOCUMENTS)} documents and {len(QUESTIONS)} questions to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

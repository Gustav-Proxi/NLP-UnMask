# UnMask — Socratic OT Anatomy Tutor

**CSE 635: NLP and Text Mining · University at Buffalo · Spring 2026**

UnMask helps OT students prepare for the **NBCOT certification exam** by teaching through questions, not answers.

---

## How it works

Instead of telling you the answer, UnMask:
1. **Retrieves** the correct answer from the OpenStax Anatomy & Physiology textbook
2. **Hides** that answer from you in a masked internal field
3. **Asks** a Socratic question calibrated to guide you toward discovering it yourself

This is called **Progressive Context Revelation (PCR)** — the system's core novelty. The answer is architecturally absent from your view until you demonstrate enough mastery to earn it.

---

## Session structure (~15 min)

| Phase | Time | What happens |
|-------|------|--------------|
| Warm-up | 0–2 min | 4 quick diagnostic questions to calibrate where you start |
| Tutoring | 2–12 min | Socratic loop — questions, hints, concept graph tracing |
| Assessment | 12–14 min | Clinical scenario — explain your reasoning in free text |
| Wrap-up | 14–15 min | Mastery summary + weak topics to review |

---

## Topics covered (MVP)

- Brachial plexus (origin → trunks → cords → terminal branches)
- Peripheral nerves: axillary, radial, median, ulnar
- Rotator cuff muscles (SITS)

---

## Debug panel

The sidebar shows live session metadata visible to the instructor:
- Retrieved chunks with PCR mode (`context_only` / `prerequisite_first` / `full_reveal`)
- Per-concept mastery scores (🔴 < 0.4 · 🟡 0.4–0.7 · 🟢 > 0.7)
- Phase, turn count, elapsed time

---

*Built by Sanika Vilas Najan & Vaishak Girish Kumar*

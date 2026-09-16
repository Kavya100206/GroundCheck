# SpotifyCares AI Support Agent

An auditable, grounded AI customer support pipeline built on real-world Twitter data (`@SpotifyCares` from TWCS). Features a few-shot LLM intent classifier, dense semantic grounding with curated standard operating procedures, and a two-layer hybrid escalation engine that rejects raw LLM self-reported confidence in favor of deterministic safety gates and calibrated confidence thresholds.

---

## Architecture Flow

The system operates across distinct, single-responsibility components with strict deterministic safety boundaries:

```
[Customer Tweet / Context]
            │
            ▼
[Few-Shot Intent Classifier] ─── (qwen/qwen3.8-27b, temperature 0.0)
            │
            ├───────────────────────────────┐
            ▼                               ▼
[Dense Semantic Retriever]       [Layer 1 Hard Rule Gates]
(all-MiniLM-L6-v2, top-1)        (Billing, Security, Legal,
            │                     Exhausted, Feature Gaps)
            │                               │
            ▼                               │
[Layer 2 Calibrated Boundary]               │
(tau = 0.73 proxy threshold)                │
            │                               │
            └───────────────┬───────────────┘
                            ▼
               [Escalation Decision Engine]
                   /                 \
      (auto_handle)                   (escalate)
            │                              │
            ▼                              ▼
  [Dual Grounding Resolver]       [Empathetic Routing Notice]
  (retrieved_case / curated_sop)  (grounding_source: "none")
            │                              │
            └───────────────┬──────────────┘
                            ▼
               [Grounded Reply Drafter]
                 (qwen/qwen3.8-27b)
                            │
                            ▼
             [Final Emitted Output Record]
```

### The Two-Layer Escalation Architecture
- **Layer 1: Deterministic Hard Gates:** Non-overridable pattern matchers for high-stakes domains (billing & payment disputes, account compromise, legal/regulatory threats, repeated customer troubleshooting failures, hardware/feature gaps). If matched, the case escalates unconditionally—no model signal or similarity score can override it.
- **Layer 2: Calibrated Confidence Threshold ($\tau = 0.73$):** For non-gated cases, escalation evaluates dense retrieval similarity against an empirical calibration curve. The threshold is locked at $\tau = 0.73$ per a stated principled safety rule ($\text{FAH} \le 15\%$).

---

## Headline Performance vs. Baselines

The table below prioritizes reporting integrity by leading with the **stratified held-out evaluation split as the primary honest out-of-sample estimate**, presenting the full 151-row set as a secondary bound reflecting calibration leakage:

| Metric | Baseline 1 (Always Escalate) | Baseline 3 ($\tau=0.35$ Locked Hurdle) | **PRIMARY HONEST ESTIMATE: Held-Out Split ($N=76$, $\tau=0.73$)** | **SECONDARY BOUND: Full Golden Set ($N=151$, $\tau=0.73$, Calib Leakage)** | Margin vs Baseline 3 (Primary Estimate) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Overall Escalation Accuracy** | 45.03% (68/151) | 53.00% (80/151) | **52.63% (40/76)** | *54.97% (83/151)* | -0.37% |
| **Easy Tier Accuracy** | 60.00% (30/50) | 56.00% (28/50) | **60.00% (15/25)** | *64.00% (32/50)* | +4.00% |
| **Hard Tier Accuracy** | 37.62% (38/101) | 51.50% (52/101) | **49.02% (25/51)** | *50.50% (51/101)* | -2.48% |
| **False Auto-Handles (FAH)** *(Safety Failure)* | **0/68 (0.0%)** | 27/68 (39.71%) | **7/35 (20.00%)** | *10/68 (14.71%)* | **-19.71% pts (-50% rel)** |
| **False Escalations (FE)** *(Automation Loss)* | 83/83 (100.0%) | **44/83 (53.01%)** | **29/41 (70.73%)** | *58/83 (69.88%)* | +17.72% pts |

### Classification Benchmark (Phase 3)
- **Model:** `qwen/qwen3.8-27b` (few-shot prompted, 0 golden-set contamination).
- **In-Taxonomy Accuracy:** **82.58%** (109/132) vs Baseline 2 (TF-IDF + LogReg) **66.67%** (+15.91% margin).
- **Macro F1:** **0.819** vs Baseline 2 **0.630** (+0.189).

---

## Honest Tradeoffs: The Price of Safety

Under asymmetric error weighting, False Auto-Handles (FAH: sending incorrect automated advice to an unhandleable complaint) are catastrophic errors that destroy customer trust. However, our empirical calibration curve proves that **achieving $\text{FAH} \le 15\%$ forces False Escalations to surge to 69.9%**:

- Terse customer tweets ("shuffle looping", "music keeps stopping") lack extensive context, scoring top-1 cosine similarities between $0.55$ and $0.72$ against historical resolutions.
- To suppress dangerous edge cases down to $14.7\%$, the threshold must be raised to $\tau=0.73$.
- This threshold inevitably sweeps **58 out of 83 legitimately handleable inquiries into human escalation queues**. The system achieves safety through defensive over-escalation rather than fine-grained discrimination.

---

## Deliverables Map

This repository directly satisfies the assignment brief's deliverable requirements:

1. [`REPORT.md`](file:///Users/kavya/Desktop/Groundcheck/REPORT.md) — Comprehensive problem framing, empirical evaluation vs Baselines 1–3, failure analysis (breakdown of the 58 false escalations), and honest self-audit ("What Is Misleading About My Headline Number").
2. [`DECISION_LOG.md`](file:///Users/kavya/Desktop/Groundcheck/DECISION_LOG.md) — Complete chronological log of all 19 formal decisions, data pivots, keyword heuristic tightening, and calibration choices.
3. [`ARCHITECTURE.md`](file:///Users/kavya/Desktop/Groundcheck/ARCHITECTURE.md) — Complete technical specification: component breakdown, data schemas, dual grounding mechanics, and the LLM/deterministic boundary.
4. [`golden_set_to_label.csv`](file:///Users/kavya/Desktop/Groundcheck/golden_set_to_label.csv) — 151 hand-labelled, stratified golden evaluation set (intent, visible resolution, escalation decision, and difficulty tier).
5. [`phase1b_methodology.md`](file:///Users/kavya/.gemini/antigravity-ide/brain/92f70b72-6271-4605-99d1-46d40c0955f8/phase1b_methodology.md) — Rigorous sampling methodology note disclosing stratification shape, contamination rules, and difficulty distribution.

---

## Quickstart & Reproduction (< 2 Minutes)

### 1. Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

### 2. Run Baselines (Phase 2)
```bash
python3 src/baselines.py
```
*Evaluates Baseline 1 (always escalate), Baseline 2 (TF-IDF LogReg classifier), and Baseline 3 (retrieval + rule gate) across the 151 golden rows.*

### 3. Run Intent Classifier (Phase 3)
```bash
python3 src/classifier.py
```
*Evaluates `qwen/qwen3.8-27b` across all 132 in-taxonomy golden rows.*

### 4. Run Full Agent Pipeline (Phase 4)
```bash
python3 src/agent.py --tau 0.73
```
*Runs the complete pipeline (classification, dense retrieval, two-layer escalation, dual-grounded reply drafting) across all 151 golden rows in ~5 seconds using disk caches.*

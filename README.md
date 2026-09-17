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

1. [`REPORT.md`](file:///Users/kavya/Desktop/Groundcheck/REPORT.md) — Comprehensive problem framing, empirical evaluation vs Baselines 1–3, Top-5 Failure Taxonomy, Phase 5 independent judge validation, and honest self-audit ("What Is Misleading About My Headline Numbers").
2. [`DECISION_LOG.md`](file:///Users/kavya/Desktop/Groundcheck/DECISION_LOG.md) — Complete chronological log of all 24 formal decisions, data pivots, keyword heuristic tightening, and calibration choices with an Executive Phase Index.
3. [`ARCHITECTURE.md`](file:///Users/kavya/Desktop/Groundcheck/ARCHITECTURE.md) — Complete technical specification: component breakdown, data schemas, dual grounding mechanics, and the LLM/deterministic boundary.
4. [`eval/metrics.py`](file:///Users/kavya/Desktop/Groundcheck/eval/metrics.py) — Automated evaluation harness computing classification F1, headline escalation accuracy, FAH, FE, and asymmetric loss ($4 \times \text{FAH} + 1 \times \text{FE}$).
5. [`eval/judge.py`](file:///Users/kavya/Desktop/Groundcheck/eval/judge.py) — Independent cross-architecture LLM judge (`openai/gpt-oss-120b`) evaluating Groundedness (1–5, zero synthetic URLs) and Tone (1–5).
6. [`eval/validate_judge.py`](file:///Users/kavya/Desktop/Groundcheck/eval/validate_judge.py) — Human-judge validation study across 50 golden rows reporting Cohen's Kappa, asymptotic standard errors, and tier disaggregation.
7. [`golden_set_to_label.csv`](file:///Users/kavya/Desktop/Groundcheck/golden_set_to_label.csv) — 151 hand-labelled, stratified golden evaluation set (intent, visible resolution, escalation decision, and difficulty tier).

---

## Quickstart & Dual-Track Reproducibility Benchmark

The codebase explicitly separates **instant cached verification** from **genuine cold-start regeneration** so reviewers know exactly what each command verifies:

### 1. Environment Setup (< 1 Minute)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 2. Verification Mode A: Instant Cached Replay (< 2 Seconds)
*Instantly recomputes and verifies all reported metrics against persisted prediction artifacts without API calls:*
```bash
# Verify classification F1, escalation accuracy, FAH/FE, and asymmetric loss
python3 -m eval.metrics

# Verify LLM judge agreement (80.0%), Cohen's Kappa (0.381), and tier breakdown
python3 -m eval.validate_judge
```

### 3. Verification Mode B: Cold-Start Full Regeneration (~2.5 to 3.5 Minutes)
*Executes the complete pipeline from scratch with zero cached predictions, live embeddings, and live LLM calls (tested wall-clock: ~3.0 min $\ll$ 15 min limit):*
```bash
# 1. Train and evaluate Baselines 1-3 (8.8s)
python3 src/baselines.py

# 2. Live few-shot classification across golden set via qwen/qwen3.8-27b (~45s)
python3 src/classifier.py

# 3. Live two-layer escalation and grounded reply drafting across 151 rows (~45s)
python3 src/agent.py --tau 0.73 --no-cache

# 4. Live independent judge evaluation across 50 rows via openai/gpt-oss-120b (~40s)
python3 -m eval.validate_judge --force
```


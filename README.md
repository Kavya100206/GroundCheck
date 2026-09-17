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
            ├─────────────────────────────────┐
            ▼                                 ▼
[Dense Semantic Retriever]         [Layer 1 Hard Rule Gates]
(all-MiniLM-L6-v2, top-1)          (Billing, Security, Legal,
            │                       Exhausted, Feature Gaps)
            ▼                                 │
[Two-Threshold Gate & Auditor]                │
├─ Sim < tau_low (0.65): Hard Escalate        │
├─ Sim in [0.65, 0.73): Auto-Handle if YES    │
├─ Sim >= tau_high (0.73): Auto-Handle / Veto │
│  [Independent Resolution Checker]           │
│  (openai/gpt-oss-120b, temperature 0.0)     │
            │                                 │
            └────────────────┬────────────────┘
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
- **Layer 2: Two-Threshold Calibrated Gate + Independent Problem-Resolution Checker (`openai/gpt-oss-120b`):** For non-gated cases, escalation uses a decoupled two-threshold policy:
  - $s < \tau_{low} = 0.65$: Unconditional escalation (saves API latency and token costs).
  - $s \in [0.65, 0.73)$: Auto-handle *only* if the independent `openai/gpt-oss-120b` checker verifies that the candidate procedure actually resolves the customer's technical symptom (recovering colloquial complaints from the dead zone).
  - $s \ge \tau_{high} = 0.73$: Passes through the resolution check; if the checker flags an unresolvable bug or procedural brush-off, it issues an immediate veto to force escalation (resolving Failure Mode 3).
  *(The original single-threshold $\tau=0.73$ design is preserved as a baseline-of-record).*

---

## Headline Performance vs. Baselines

The table below prioritizes reporting integrity by leading with the **stratified held-out evaluation split as the primary honest out-of-sample estimate**, presenting the delivered two-threshold architecture alongside the original Phase 4 baseline:

| Metric | Baseline 1 (Always Escalate) | Baseline 3 ($\tau=0.35$ Locked Hurdle) | Phase 4 Baseline (Held-Out Split, $\tau=0.73$) | **DELIVERED ARCHITECTURE: Held-Out Split ($N=76$, $\tau \in [0.65, 0.73]$ + Check)** | Secondary Bound (Full $N=151$, $\tau=0.73$, Calib Leakage) | Margin vs Phase 4 Baseline (Held-Out) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall Escalation Accuracy** | 45.03% (68/151) | 53.00% (80/151) | 52.63% (40/76) | **64.47% (49/76)** | *54.97% (83/151)* | **+11.84% pts** |
| **Easy Tier Accuracy** | 60.00% (30/50) | 56.00% (28/50) | 60.00% (15/25) | **68.00% (17/25)** | *64.00% (32/50)* | **+8.00% pts** |
| **Hard Tier Accuracy** | 37.62% (38/101) | 51.50% (52/101) | 49.02% (25/51) | **62.75% (32/51)** | *50.50% (51/101)* | **+13.73% pts** |
| **False Auto-Handles (FAH)** *(Safety Failure)* | **0/68 (0.0%)** | 27/68 (39.71%) | 7/35 (20.00%) | **2/35 (5.71%)** | *10/68 (14.71%)* | **-14.29% pts (-71.4% rel)** |
| **False Escalations (FE)** *(Automation Loss)* | 83/83 (100.0%) | **44/83 (53.01%)** | 29/41 (70.73%) | **25/41 (60.98%)** | *58/83 (69.88%)* | **-9.75% pts (-13.8% rel)** |
| **Asymmetric Cost ($4 \times \text{FAH} + 1 \times \text{FE}$)** | 83.0 | 152.0 | 57.0 | **33.0** | *98.0* | **-24.0 (-42.1% rel)** |

### Classification Benchmark (Phase 3)
- **Model:** `qwen/qwen3.8-27b` (few-shot prompted, 0 golden-set contamination).
- **In-Taxonomy Accuracy:** **82.58%** (109/132) vs Baseline 2 (TF-IDF + LogReg) **66.67%** (+15.91% margin).
- **Macro F1:** **0.819** vs Baseline 2 **0.630** (+0.189).

---

## Honest Tradeoffs: The Evolution from Scalar Thresholds to Semantic Verification

Under asymmetric error weighting ($w_{FAH}=4.0, w_{FE}=1.0$), False Auto-Handles (sending incorrect automated advice to an unhandleable complaint) are catastrophic errors that destroy customer trust. 

In the initial Phase 4 design, relying strictly on a single scalar cosine similarity cutoff ($\tau = 0.73$) produced an acute operational penalty: **False Escalations surged to 69.9% (70.73% on held-out data)**, routing 7 out of 10 handleable inquiries to human queues. Terse customer tweets ("shuffle looping", "music keeps stopping") scored top-1 similarities between $0.55$ and $0.72$ against historical resolutions, falling into a dead zone where a scalar number could not distinguish genuine colloquial queries from unresolvable bugs.

However, the delivered post-Phase-6 architecture proved that **this automation collapse was specifically an artifact of single-scalar similarity thresholding, not an inescapable operational law**:
- By introducing a decoupled two-threshold gate with independent LLM semantic verification (`openai/gpt-oss-120b`), queries in the $[0.65, 0.73)$ dead zone are evaluated on whether the candidate procedure actually resolves the customer's technical symptom.
- On the frozen held-out split ($N=76$), this semantic verification reduced False Auto-Handles down to **5.71%** (well below the 15% safety requirement) while simultaneously cutting False Escalations to **60.98%** and reducing asymmetric error cost by **42.1%** (from 57.0 down to 33.0). Precision discrimination requires semantic verification rather than scalar cutoff tuning.

---

## Deliverables Map

This repository directly satisfies the assignment brief's deliverable requirements:

1. [`REPORT.md`](file:///Users/kavya/Desktop/Groundcheck/REPORT.md) — Comprehensive problem framing, empirical evaluation vs Baselines 1–3, Top-5 Failure Taxonomy, Phase 5 independent judge validation, post-Phase-6 two-threshold architecture results (§6.3.1), and honest self-audit ("What Is Misleading About My Headline Numbers").
2. [`DECISION_LOG.md`](file:///Users/kavya/Desktop/Groundcheck/DECISION_LOG.md) — Complete chronological log of all 30 formal decisions, data pivots, keyword heuristic tightening, and calibration choices with an Executive Phase Index.
3. [`ARCHITECTURE.md`](file:///Users/kavya/Desktop/Groundcheck/ARCHITECTURE.md) — Complete technical specification: component breakdown, data schemas, dual grounding mechanics, and the LLM/deterministic boundary.
4. [`eval/metrics.py`](file:///Users/kavya/Desktop/Groundcheck/eval/metrics.py) — Automated evaluation harness computing classification F1, headline escalation accuracy, FAH, FE, and asymmetric loss ($4 \times \text{FAH} + 1 \times \text{FE}$).
5. [`eval/judge.py`](file:///Users/kavya/Desktop/Groundcheck/eval/judge.py) — Independent cross-architecture LLM judge (`openai/gpt-oss-120b`) evaluating Groundedness (1–5, zero synthetic URLs) and Tone (1–5).
6. [`eval/validate_judge.py`](file:///Users/kavya/Desktop/Groundcheck/eval/validate_judge.py) — Human-judge validation study across 50 golden rows reporting Cohen's Kappa, asymptotic standard errors, and tier disaggregation.
7. [`golden_set_to_label.csv`](file:///Users/kavya/Desktop/Groundcheck/golden_set_to_label.csv) — 151 hand-labelled, stratified golden evaluation set (intent, visible resolution, escalation decision, and difficulty tier).
8. [`src/resolution_check.py`](file:///Users/kavya/Desktop/Groundcheck/src/resolution_check.py) — Independent cross-architecture problem-resolution verification module (`openai/gpt-oss-120b`) with persistent disk cache (`src/resolution_check_cache.json`), auditing candidate procedures inline before auto-handling.

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

### 3. Verification Mode B: Cold-Start Full Regeneration (~7 to 8 Minutes Staged / ~24 Minutes Standalone)
*Executes the complete pipeline from scratch with zero cached predictions, live embeddings, and live multi-model LLM calls:*
```bash
# 1. Train and evaluate Baselines 1-3 (8.8s)
python3 src/baselines.py

# 2. Live few-shot classification across golden set via qwen/qwen3.8-27b (~45s)
python3 src/classifier.py

# 3. Live two-layer escalation, inline gpt-oss-120b resolution verification, and reply drafting across 151 rows
# Runs the delivered two-threshold architecture with gpt-oss-120b verification by default:
# (Tested wall-clock: ~5.5 min reusing Step 2 classifier output; 1425.4s / ~23.8 min when running standalone from cold scratch)
python3 src/agent.py --no-cache

# Optional: reproduce the historical Phase 4 single-threshold baseline (tau=0.73 without LLM check, ~45s):
# python3 src/agent.py --no-check --no-cache

# 4. Live independent judge evaluation across 50 rows via openai/gpt-oss-120b (~40s)
python3 -m eval.validate_judge --force
```


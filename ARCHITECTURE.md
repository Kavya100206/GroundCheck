# ARCHITECTURE.md — AI Customer Support Agent: Technical Architecture

This document defines the complete technical architecture, data flow, component boundaries, schema contracts, and testing discipline for the SpotifyCares AI customer support agent.

---

## 1. Data Flow

The agent processes incoming customer inquiries through an end-to-end, multi-stage pipeline:

```
[Inbound Tweet / Thread Context]
               │
               ▼
   [Few-Shot LLM Intent Classifier] (qwen/qwen3.8-27b)
               │
               ├──────────────────────────────┐
               ▼                              ▼
  [Dense Semantic Retriever]       [Layer 1 Hard Rule Gates]
  (all-MiniLM-L6-v2, top-1)        (Billing, Security, Legal,
               │                    Exhausted, Feature Gap)
               │                              │
               ▼                              │
  [Layer 2 Calibrated Boundary]               │
  (tau = 0.73 proxy threshold)                │
               │                              │
               └──────────────┬───────────────┘
                              ▼
                 [Escalation Decision Engine]
                     /                 \
        (auto_handle)                   (escalate)
              │                              │
              ▼                              ▼
    [Dual Grounding Resolver]       [Empathetic Routing Notice]
    (retrieved_case / curated_sop)  (grounding_source: "none")
              │                              │
              └──────────────┬───────────────┘
                             ▼
                 [Grounded Reply Drafter]
                   (qwen/qwen3.8-27b)
                             │
                             ▼
              [Final Emitted Output Record]
```

1. **Ingest**: Raw customer tweet text (+ thread context for multi-turn interactions) is loaded.
2. **Classify**: Customer text is passed to the few-shot prompted LLM classifier (`qwen/qwen3.8-27b`), returning one of the 5 locked taxonomy intents.
3. **Retrieve / Ground**: Using the predicted intent and message text, the dense semantic retriever searches the strict 8,310 visible-resolution corpus (`all-MiniLM-L6-v2`) for top-1 similar historical cases.
4. **Escalation Decision**:
   - Case is evaluated against Layer 1 deterministic rule gates first (billing, security, legal, exhausted troubleshooting, feature gap). If matched, it escalates unconditionally.
   - If no hard gate fires, Layer 2 evaluates retrieval similarity against the calibrated operational threshold ($\tau = 0.73$). If $s \ge 0.73$, it auto-handles; otherwise, it escalates.
5. **Reply Drafting**: Conditioned on the decision and grounding evidence:
   - If `auto_handle`: resolves grounding through a dual mechanism (direct `retrieved_case` if concrete public steps exist, or `curated_sop` from `src/policy.py`). The LLM drafts an empathetic, procedural Twitter reply (<250 chars, zero synthetic URLs).
   - If `escalate`: generates an empathetic routing acknowledgment informing the customer their issue is being transferred to a specialist team, with zero troubleshooting advice.
6. **Output**: An auditable output record is emitted adhering to the schema below.

---

## 2. Component Breakdown

### 2.1 Intent Classifier ([`src/classifier.py`](file:///Users/kavya/Desktop/Groundcheck/src/classifier.py))
- **Responsibility:** Map customer text to exactly one of the 5 committed taxonomy labels (`playback_issue`, `app_technical`, `account_login`, `device_platform`, `premium_billing`).
- **Implementation:** Few-shot prompted chat completions via `qwen/qwen3.8-27b` (temperature 0.0). Examples drawn exclusively from the unsampled training pool (zero golden-set leakage).
- **Boundary:** Single responsibility; does not decide escalation or draft content.

### 2.2 Grounding Retriever ([`src/retrieval.py`](file:///Users/kavya/Desktop/Groundcheck/src/retrieval.py))
- **Responsibility:** Return the most semantically relevant historical resolution from the visible-resolution corpus.
- **Implementation:** Dense embedding search using `all-MiniLM-L6-v2` over 8,310 verified visible-resolution historical threads (~21.2% of unsampled training pool). Retrieval is intent-scoped to minimize cross-domain false matches.
- **Boundary:** Supplies evidence and cosine similarity proxy signal; does not draft replies.

### 2.3 Curated Policy Reference ([`src/policy.py`](file:///Users/kavya/Desktop/Groundcheck/src/policy.py))
- **Responsibility:** Provide verifiable procedural grounding when historical retrieval matches diagnostic questions, DM deflections, or status updates (~78% of traffic).
- **Implementation:** 17 canonical SOPs extracted from 1,173 concrete actionable threads. Contains explicit step sequences and zero synthetic URLs (relying strictly on in-app UI paths: `Settings > Storage > Clear Cache`, `Log out > Restart > Log in`, etc.).
- **Boundary:** Deterministic domain fallback; ensures replies are never ungrounded.

### 2.4 Escalation Decision Engine ([`src/escalation.py`](file:///Users/kavya/Desktop/Groundcheck/src/escalation.py))
- **Responsibility:** Decide `auto_handle` vs `escalate` with an explicit reason string and gate attribution.
- **Implementation:** Two-layer hybrid architecture. Layer 1 hard regex gates + Layer 2 calibrated threshold ($\tau=0.73$). Highest-consequence decision in the pipeline; built as deterministic code.

### 2.5 Reply Drafter ([`src/drafter.py`](file:///Users/kavya/Desktop/Groundcheck/src/drafter.py))
- **Responsibility:** Draft concise, empathetic customer-facing Twitter replies strictly constrained by verified evidence.
- **Implementation:** Chat completion via `qwen/qwen3.8-27b` (temperature 0.0). Conditioned on query, intent, decision, and evidence. Enforces Twitter length (<250 chars) and strictly bans hallucinated URLs.

### 2.6 Evaluation Harness (Planned Phase 5)
- **Responsibility:** Independent verification module evaluating classifier accuracy/F1, escalation metrics with asymmetric weighting, groundedness audit, and LLM-as-a-judge agreement across difficulty tiers.

---

## 3. LLM / Deterministic-Code Boundary (Core Architectural Decision)

This boundary is treated as a first-class architectural principle rather than an implementation detail:

- **Entrusted to the LLM:**
  - Semantic intent classification (language nuances, colloquial expressions, customer frustration).
  - Reply generation (Twitter tone, empathy, formatting, procedural synthesis).
- **Enforced Deterministically in Code (Never Delegated to the LLM):**
  - **Self-Reported Confidence Rejected:** LLM self-reported confidence is known to be poorly calibrated, sycophantic, and prone to over-confidence on hallucinations. It is completely rejected as an escalation signal.
  - **Non-Overridable Layer 1 Hard Gates:** High-stakes domains (unauthorized charges, payment disputes, hacked accounts, legal threats, repeated customer troubleshooting failures, hardware feature gaps) trigger unconditional escalation via deterministic pattern matching. No retrieval similarity score or model output can override a hard gate.
  - **Calibrated Layer 2 Boundary:** For non-gated cases, escalation is determined by an empirically calibrated similarity threshold ($\tau = 0.73$) derived against golden-set correctness, satisfying `build.md`'s safety hurdle ($\text{FAH} \le 15\%$).

---

## 4. Data Schemas

### Golden Evaluation Record (`golden_set_to_label.csv`)
```json
{
  "id": "gs_0001",
  "customer_tweet_id": "115858_12345",
  "customer_text": "can you add a button to play a random song...",
  "brand_tweet_id": "115858_12346",
  "brand_reply_text": "Hey there! You can submit feature requests on our Community...",
  "source": "sampled_pool",
  "sampling_stratum": "playback_issue_easy",
  "gold_intent": "other",
  "has_visible_resolution": false,
  "gold_decision": "escalate",
  "gold_decision_reason": "Feature request / wishlist inquiry requiring product team feedback",
  "difficulty_tier": "easy"
}
```

### Agent Prediction Output Record (`src/agent_predictions_golden151.csv`)
```json
{
  "id": "gs_0004",
  "predicted_intent": "playback_issue",
  "drafted_reply": "We hear you! To fix the shuffle issue, please log out, fully close the app, restart your device, and log back in. This often resolves playback glitches. Let us know if it helps!",
  "decision": "auto_handle",
  "decision_reason": "High grounding confidence (similarity 0.766 >= threshold 0.73); standard troubleshooting applicable",
  "grounding_source": "curated_sop",
  "evidence_used": ["Curated SOP [PB_01_BASIC_REFRESH]: Quick Session & Cache Refresh Protocol"],
  "gate_triggered": null,
  "calibrated_confidence": 0.766,
  "difficulty_tier": "easy",
  "gold_decision": "auto_handle",
  "gold_intent": "playback_issue"
}
```

### Judge Output Record (Phase 5 Schema)
```json
{
  "id": "gs_0004",
  "groundedness_score": 5,
  "tone_score": 5,
  "judge_decision": "pass",
  "judge_notes": "Reply strictly reflects SOP PB_01 logout/restart sequence with zero hallucinated links.",
  "difficulty_tier": "easy"
}
```

---

## 5. Reproducibility Mechanics (< 15 Minute Benchmark)

- **Strict Subsample Scope:** Pipeline operates over a 151-row golden set and an 8,310-document visible-resolution index, eliminating the multi-million row TWCS loading bottleneck.
- **Offline Model Caching:** SentenceTransformers runs with `HF_HUB_OFFLINE=1`, loading cached `all-MiniLM-L6-v2` embeddings in <1 second without network retries.
- **Smart Result Caching:** Classifier predictions and drafted replies are persisted incrementally to disk (`src/classifier_predictions.csv`, `src/agent_predictions_golden151.csv`), enabling subsequent evaluation sweeps to execute in ~5 seconds.
- **Deterministic Seeding:** Stratified sampling and calibration splits use explicit seeds (`random_state=42`).

---

## 6. Testing Strategy

1. **Fast Deterministic Unit Tests (Zero API Calls, <2s runtime):**
   - Schema validation for golden set and prediction output records.
   - Non-overridable adversarial safety check: 5 hand-crafted high-stakes prompts (double charge, Russian account hack, lawsuit threat, 3x reinstall crash, iPhone X display) assert unconditional escalation even at $s=0.99$.
   - Operational threshold lookup logic and gate regex assertion.
2. **End-to-End Live LLM Tests:**
   - Evaluates full pipeline over golden set rows via pinned model `qwen/qwen3.8-27b` at temperature 0.0.
   - Monitors per-intent classification F1, headline escalation accuracy, FAH, and FE rates.

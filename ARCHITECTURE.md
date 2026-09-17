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
               ├────────────────────────────────┐
               ▼                                ▼
  [Dense Semantic Retriever]         [Layer 1 Hard Rule Gates]
  (all-MiniLM-L6-v2, top-1)          (Billing, Security, Legal,
               │                      Exhausted, Feature Gap)
               ▼                                │
  [Two-Threshold Gate & Auditor]                │
  ├─ Sim < tau_low (0.65): Hard Escalate        │
  ├─ Sim in [0.65, 0.73): Auto-Handle if YES    │
  ├─ Sim >= tau_high (0.73): Auto-Handle / Veto │
  │  [Independent Resolution Checker]           │
  │  (openai/gpt-oss-120b, temperature 0.0)     │
               │                                │
               └────────────────┬───────────────┘
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

1. **Ingest**: Raw customer tweet text (+ thread context for multi-turn interactions) is loaded.
2. **Classify**: Customer text is passed to the few-shot prompted LLM classifier (`qwen/qwen3.8-27b`), returning one of the 5 locked taxonomy intents.
3. **Retrieve / Ground**: Using the predicted intent and message text, the dense semantic retriever searches the strict 8,310 visible-resolution corpus (`all-MiniLM-L6-v2`) for top-1 similar historical cases.
4. **Escalation Decision**:
   - Case is evaluated against Layer 1 deterministic rule gates first (billing, security, legal, exhausted troubleshooting, feature gap). If matched, it escalates unconditionally.
   - If no hard gate fires, Layer 2 evaluates retrieval similarity against a decoupled two-threshold gate ($\tau_{low}=0.65, \tau_{high}=0.73$) coupled with the independent Problem-Resolution Checker (`openai/gpt-oss-120b`):
     - $s < \tau_{low} = 0.65$: Unconditional escalation (saves API latency and token costs).
     - $s \in [0.65, 0.73)$: Auto-handle *only* if the independent resolution checker verifies that the candidate procedure actually resolves the customer's specific technical symptom (recovering colloquial queries from the dead zone).
     - $s \ge \tau_{high} = 0.73$: Candidate accepted for auto-handling unless the resolution checker detects an unresolvable bug or procedural brush-off, in which case it issues an immediate hard veto (`resolution_check_veto`).
   *(The original Phase 4 single-threshold $\tau=0.73$ scalar cutoff is preserved in code via `--no-check` as a baseline-of-record).*
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
- **Responsibility:** Decide `auto_handle` vs `escalate` with an explicit reason string, gate attribution, and decision layer tagging.
- **Implementation:** Two-layer hybrid architecture. Layer 1 deterministic regex gates + Layer 2 calibrated two-threshold boundary ($\tau_{low}=0.65, \tau_{high}=0.73$) coupled with the independent resolution checker. Built as pure deterministic control logic in code; no LLM self-reported confidence is permitted to decide routing.

### 2.5 Independent Resolution Checker ([`src/resolution_check.py`](file:///Users/kavya/Desktop/Groundcheck/src/resolution_check.py))
- **Responsibility:** Audit candidate procedures inline before auto-handling to verify that the retrieved SOP or historical resolution directly and practically resolves the customer's specific technical symptom, preventing procedural brush-offs (FM3) and safely recovering colloquial complaints from the dead zone (FM2).
- **Implementation:** Pinned chat completions via `openai/gpt-oss-120b` (temperature 0.0, 750 max tokens). Deliberately utilizes an independent, larger cross-architecture model family from the drafter/classifier (`qwen/qwen3.8-27b`) to eliminate correlated blind spots and avoid self-grading bias (following the precedent established in Decision Log Entry 20 for the Phase 5 judge). Employs persistent disk caching (`src/resolution_check_cache.json`) for instant, deterministic replayability.
- **Boundary:** Inline escalation auditor; emits boolean `resolves_problem`, confidence score, and audit reason string. Does not draft customer replies or modify retrieval indices.

### 2.6 Reply Drafter ([`src/drafter.py`](file:///Users/kavya/Desktop/Groundcheck/src/drafter.py))
- **Responsibility:** Draft concise, empathetic customer-facing Twitter replies strictly constrained by verified evidence.
- **Implementation:** Chat completion via `qwen/qwen3.8-27b` (temperature 0.0). Conditioned on query, intent, decision, and evidence. Enforces Twitter length (<250 chars), validates against synthetic URL hallucination via regex guardrails, and incorporates automated single-retry fallback on formatting failure.

### 2.7 Evaluation Harness ([`eval/metrics.py`](file:///Users/kavya/Desktop/Groundcheck/eval/metrics.py), [`eval/judge.py`](file:///Users/kavya/Desktop/Groundcheck/eval/judge.py), [`eval/validate_judge.py`](file:///Users/kavya/Desktop/Groundcheck/eval/validate_judge.py))
- **Responsibility:** Multi-layer evaluation suite computing classifier accuracy/F1, escalation metrics with asymmetric cost weighting ($4 \times \text{FAH} + 1 \times \text{FE}$), human-aligned groundedness/tone auditing via `openai/gpt-oss-120b`, and inter-annotator agreement (Cohen's Kappa) across difficulty tiers.

---

## 3. LLM / Deterministic-Code Boundary (Core Architectural Decision)

This boundary is treated as a first-class architectural principle rather than an implementation detail:

- **Entrusted to the LLMs:**
  - **Semantic Intent Classification:** Handled by `qwen/qwen3.8-27b` (temperature 0.0) to parse colloquial customer language, multi-turn thread context, and implicit user frustration.
  - **Semantic Problem-Resolution Verification:** Handled by `openai/gpt-oss-120b` (temperature 0.0) to evaluate whether a proposed procedure actually resolves the customer's technical symptom or merely represents an unresolvable bug or procedural brush-off. Crucially, this audit is performed by an **independent, decoupled model family** rather than having the drafter grade its own candidate.
  - **Reply Generation:** Handled by `qwen/qwen3.8-27b` (temperature 0.0) to synthesize concise, empathetic Twitter copy strictly grounded in verified evidence.
- **Enforced Deterministically in Code (Never Delegated to the LLM):**
  - **Self-Reported Confidence Rejected:** Raw LLM self-reported confidence is known to be uncalibrated, sycophantic, and prone to over-confidence on hallucinations. It is strictly rejected as an escalation signal.
  - **Non-Overridable Layer 1 Hard Gates:** High-stakes domains (unauthorized charges, payment disputes, compromised accounts, legal threats, repeated customer troubleshooting failures, hardware/feature gaps) trigger unconditional escalation via deterministic regex pattern matching. No retrieval similarity score or model output can override a hard gate.
  - **Decoupled Two-Threshold Control Logic:** The routing decision flow is strictly governed by deterministic Python logic in `src/escalation.py`:
    1. *Below $\tau_{low} = 0.65$ (Hard Escalate):* Unconditional escalation enforced in code. No resolution checker call is initiated, saving API latency and token budget on obvious edge cases.
    2. *Moderate Band $\tau_{low} \le s < \tau_{high}$ ($[0.65, 0.73)$, Conditional Recovery):* The agent is permitted to auto-handle *only if* the independent `openai/gpt-oss-120b` checker returns `resolves_problem: true`. If the checker returns `false`, times out, or fails to parse, deterministic code defaults to safe escalation.
    3. *High Confidence Band $s \ge \tau_{high} = 0.73$ (Pass with Hard Veto):* Retrieval similarity is sufficiently high for auto-handling, but code enforces that if the independent checker returns `resolves_problem: false` (detecting a procedural brush-off or unresolvable bug), code issues an immediate non-overridable `resolution_check_veto` and forces human escalation.
    *(The historical single-threshold scalar cutoff $\tau=0.73$ is maintained in code via `--no-check` as the Phase 4 baseline-of-record).*

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
  "decision_reason": "Moderate grounding confidence (similarity 0.682 >= 0.65) verified by resolution check: Candidate procedure provides standard cache clear protocol which addresses playlist loading glitches",
  "grounding_source": "curated_sop",
  "evidence_used": ["Curated SOP [PB_01_BASIC_REFRESH]: Quick Session & Cache Refresh Protocol"],
  "gate_triggered": null,
  "calibrated_confidence": 0.682,
  "difficulty_tier": "easy",
  "gold_decision": "auto_handle",
  "gold_intent": "playback_issue"
}
```
*Note on Schema Extensions:*
- When the independent resolution checker triggers an override at $s \ge 0.73$, `gate_triggered` is populated with `"resolution_check_veto"` and `decision_reason` records the checker's diagnostic rationale (`"Procedural brush-off veto (resolution check failed): ..."`).
- When a moderate-similarity case in $[0.65, 0.73)$ is verified, `decision_reason` explicitly records the resolution check confirmation string, preserving full traceability while remaining 100% backward-compatible with downstream evaluation scripts.

### Resolution Check Cache Record (`src/resolution_check_cache.json`)
```json
{
  "hash_key_128bit": {
    "resolves_problem": true,
    "confidence": 0.85,
    "reason": "The SOP outlines clearing the local cache and logging out/in, which addresses corrupted client playback queues.",
    "model": "openai/gpt-oss-120b",
    "timestamp": 1726554600.0
  }
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

## 5. Reproducibility Mechanics

- **Strict Subsample Scope:** Pipeline operates over a 151-row golden set and an 8,310-document visible-resolution index, eliminating the multi-million row TWCS loading bottleneck.
- **Offline Model Caching:** SentenceTransformers runs with `HF_HUB_OFFLINE=1`, loading cached `all-MiniLM-L6-v2` embeddings in <1 second without network retries.
- **Smart Result Caching:** Classifier predictions, drafted replies, and resolution checker verdicts are persisted incrementally to disk (`src/classifier_predictions.csv`, `src/agent_predictions_golden151.csv`, `src/resolution_check_cache.json`), enabling subsequent evaluation sweeps to execute in ~2 seconds.
- **Deterministic Seeding:** Stratified sampling and calibration splits use explicit seeds (`random_state=42`).

---

## 6. Testing Strategy

The test harness is organized into two distinct tiers:

1. **Fast Deterministic & Offline Tier (Zero Network Calls, <2s runtime):**
   - **Schema Contracts:** Validates structure, nullability, and types across golden set, prediction outputs, and resolution check cache entries.
   - **Adversarial Safety Invariants:** 5 hand-crafted high-stakes prompts (double billing charge, Russian account takeover, lawsuit threat, 3x reinstall crash, iPhone X display compatibility) assert unconditional human escalation even if similarity is artificially forced to $s=0.99$.
   - **Resolution Checker Offline Regression:** With persistent caching (`src/resolution_check_cache.json`), resolution check evaluations execute as zero-latency local dict lookups, enabling deterministic regression testing of the two-threshold decision engine (`src/escalation.py`) in CI without live API keys.
   - **URL Guardrail Unit Verification:** Asserts that reply drafting regex filters fail any drafted response containing hallucinated synthetic links not present in evidence.
2. **End-to-End Live LLM Tier:**
   - **Mode B (Default Cold-Start Subsample, <15 min):** Evaluates live pipeline over a stratified 38-row golden subsample via `python3 src/agent.py --no-cache`, verifying multi-model completions, tools, and two-threshold routing end-to-end well within assignment limits.
   - **Mode C (Full Cold-Start Regeneration, ~24 min):** Exhaustively evaluates all 151 golden set rows and 50 judge evaluations via `python3 src/agent.py --full --no-cache` and `python3 -m eval.validate_judge --force` (measured standalone wall-clock: 1425.4s / 23.8 min).

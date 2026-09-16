# Evaluation Report — AI Customer Support Agent (SpotifyCares)

## 1. Problem Statement & Problem Framing

Customer support on Twitter is fast, adversarial, and public. Within that noisy stream is a core set of recurring issues that the brand has solved many times before in a consistent manner. This project builds and evaluates an end-to-end AI support agent for `@SpotifyCares` that:
1. Classifies customer inquiries into a locked 5-intent taxonomy.
2. Drafts empathetic, concise Twitter replies strictly grounded in verified technical procedures (with zero synthetic URLs).
3. Executes a two-layer escalation decision (deterministic safety gates + calibrated confidence boundary) that refuses to trust raw LLM self-reported confidence.

The underlying challenge is demonstrating trustworthiness to a skeptical reviewer: proving where the system succeeds, establishing honest baselines, and documenting exactly where and why the system fails.

---

## 2. Brand Selection & Empirical Filtering

**Selected Brand: `SpotifyCares`.** Chosen based on real computed metrics from `twcs.csv` (2,811,774 rows):

| Brand | Unique Threads | Total Inbound | Avg Thread Length | Outbound Replies | DM-Redirect Rate | Est. Visible-Resolution Rate |
|---|---|---|---|---|---|---|
| **SpotifyCares** | 31,213 raw | 15,101 | 1.87 turns | 43,272 | 22.6% | **72.0%**† |
| AmazonHelp | 98,875 | 100,505 | 2.73 turns | 169,845 | 3.4% | 82.6%* |
| AppleSupport | 83,470 | 36,660 | 1.72 turns | 106,863 | 30.8% | 64.4% |

- *AmazonHelp Rejection:* Structurally fails intent diversity—spans shipping, physical returns, groceries, third-party sellers, Prime, and hardware, making a 4–6 intent taxonomy vacuous. Deflection pattern is phone/email redirects, making the visible-resolution heuristic misleading.
- *AppleSupport Rejection:* Highest DM-redirect rate (30.8%). Non-DM outbound replies are largely link-drops to external articles (`support.apple.com/...`) rather than in-thread troubleshooting, providing poor grounding evidence.
- †*SpotifyCares Audited Reality:* The 72.0% heuristic figure measured text length ($\ge 80$ chars) and absence of DM phrases. Rigorous hand-labelling of the golden set measured the **true concrete in-thread resolution rate at 22.0% (33/150)**; the remaining 78.0% consist of DM redirects, acknowledgments, or diagnostic questions. See Groundedness Caveat below.
- *Usable Pool:* An alternate join (`in_response_to_tweet_id` → customer root) recovered 28,576 additional English threads, establishing an effective usable pool of **~37,267 threads** (Decision Log Entry 6). Non-English traffic (~4%) was screened out.

---

## 3. Intent Taxonomy: Approach & Rationale

A hard cap of 5 intents was locked post-Phase 1 based on empirical brand traffic patterns:
1. **`playback_issue`** (Audio stops, skips, buffers, stuttering, offline playback failures).
2. **`account_login`** (Password reset, locked accounts, credentials, 14-day travel limits).
3. **`premium_billing`** (Unexpected charges, subscription cancellation, Family Plan address verification).
4. **`app_technical`** (App crashes on launch, freezes, blank screens, clean reinstall protocols).
5. **`device_platform`** (OS compatibility, Bluetooth pairing, CarPlay/Android Auto, smart TVs).

*Scope Policy on Out-of-Taxonomy Rows (Option A, Entry 13):* The 19 `other` golden rows (wishlist requests, feature suggestions) are excluded from intent classification metrics (as the 5-way classifier cannot predict `other`), but **kept in all escalation evaluations** to verify that the agent safely routes out-of-scope inquiries.

---

## 4. Definition of "Good" for this System

Quality is decomposed into three independent axes:
1. **Classification Correctness:** Evaluated per-intent via precision, recall, and macro F1, preventing dominant intents from concealing weaknesses in rarer classes.
2. **Reply Groundedness:** Factual alignment with attested brand procedures. Fluently written hallucinations are treated as failures.
   - *Groundedness Caveat:* Because only 22.0% of threads feature standalone public resolutions, 78% of handleable queries rely on **`CuratedPolicyReference`** (17 SOPs extracted from 1,173 concrete threads) rather than direct tweet-to-tweet matching.
3. **Escalation-Decision Quality (Asymmetric Error Weighting):** False Auto-Handles (FAH: sending incorrect automated advice to an unhandleable complaint) are treated as substantially more costly than False Escalations (FE: routing a handleable query to human queues), as FAH creates brand, legal, and account security risk.

---

## 5. What Was Deliberately Chosen Not to Build

- **Multi-lingual support:** Restricted strictly to English traffic.
- **Real-time streaming ingestion:** Subsample-based pipeline to guarantee <15 minute reproducibility.
- **Model fine-tuning:** Few-shot prompting and dense retrieval were prioritized for explainability and live auditability.
- **Synthetic URL generation:** Stripped all synthesized external support URLs from SOPs and banned link generation in drafter prompts.

---

## 6. Empirical Results vs. Baselines

### 6.1 Baseline Performance Summary (Locked in Phase 2)
- **Baseline 1 (Trivial Floor):** Majority-class intent achieved **39.39%** accuracy. Always-escalate policy achieved **45.03%** accuracy, setting the absolute safety floor: **0.0% FAH (0/68)**, **100.0% FE (83/83)**.
- **Baseline 2 (Classification-Only):** Word + bigram TF-IDF Logistic Regression trained on 1,250 balanced silver tweets achieved **66.67%** in-taxonomy accuracy, macro F1 = **0.630**, weighted F1 = **0.686**.
- **Baseline 3 (Simple Competitive Retrieval + Hard Gates):** Top-1 TF-IDF nearest neighbor with deterministic gates. At the locked hurdle ($\tau = 0.35$), achieved **53.00%** escalation accuracy, **39.71% FAH (27/68)**, and **53.01% FE (44/83)**.

### 6.2 Phase 3: Few-Shot Intent Classification (`qwen/qwen3.8-27b`)
Evaluated across all 132 in-taxonomy golden set rows using pure chat completions (temperature 0.0, zero golden set leakage):
- **Overall Accuracy:** **82.58%** (109/132), decisively beating Baseline 2's hurdle of **66.67%** (+15.91 percentage points).
- **Macro F1:** **0.819** (vs Baseline 2's 0.630, +0.189).
- **Weighted F1:** **0.830** (vs Baseline 2's 0.686, +0.144).
- **Per-Intent F1s:** `account_login` = 0.929 (26/27), `premium_billing` = 0.913 (21/23), `playback_issue` = 0.805 (35/52), `device_platform` = 0.765 (13/15), `app_technical` = 0.683 (14/15).
- **Hard Tier Robustness:** **83.84%** accuracy (83/99).

### 6.3 Phase 4: Escalation Decision Performance (Principled $\tau = 0.73$)

Reporting structurally leads with the **Held-Out Evaluation Split ($N=76$) as the primary honest estimate**, presenting the full 151-row figure second as an optimistic upper bound reflecting calibration leakage:

| Metric | Baseline 1 (Always Escalate) | Baseline 3 ($\tau=0.35$ Locked Hurdle) | **PRIMARY HONEST ESTIMATE: Held-Out Split ($N=76$, $\tau=0.73$)** | **SECONDARY BOUND: Full Golden Set ($N=151$, $\tau=0.73$, Calib Leakage)** | Margin vs Baseline 3 (Primary Estimate) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Overall Escalation Accuracy** | 45.03% (68/151) | 53.00% (80/151) | **52.63% (40/76)** | *54.97% (83/151)* | -0.37% |
| **Easy Tier Accuracy** | 60.00% (30/50) | 56.00% (28/50) | **60.00% (15/25)** | *64.00% (32/50)* | +4.00% |
| **Hard Tier Accuracy** | 37.62% (38/101) | 51.50% (52/101) | **49.02% (25/51)** | *50.50% (51/101)* | -2.48% |
| **False Auto-Handles (FAH)** *(Safety Failure)* | **0/68 (0.0%)** | 27/68 (39.71%) | **7/35 (20.00%)** | *10/68 (14.71%)* | **-19.71% pts (-50% rel)** |
| **False Escalations (FE)** *(Automation Loss)* | 83/83 (100.0%) | **44/83 (53.01%)** | **29/41 (70.73%)** | *58/83 (69.88%)* | +17.72% pts |

### 6.4 Grounding Source & Gate Distribution ($N=151$)
- **Grounding Distribution:** `none` (escalated): 116 (76.8%), `curated_sop`: 33 (21.9%), `retrieved_case`: 2 (1.3%).
- **Layer 1 Gate Precision:** `feature_or_compatibility_gap`: 9 (100%), `billing_dispute`: 5 (80%), `exhausted_troubleshooting`: 3 (100%), `account_security`: 2 (100%).
- **Adversarial Acceptance Check:** 5/5 hand-crafted adversarial messages (billing dispute, account hack, lawsuit threat, 3x reinstall crash, iPhone X display) escalated unconditionally (100% pass).

---

## 7. Failure Analysis: The 58 False Escalation Cases

Auditing the 58 handleable queries falsely routed to escalation at $\tau = 0.73$ highlights the operational price of safety:

| Intent Category | Handleable Rows ($N=83$) | Falsely Escalated (FE) | FE Rate within Intent | Primary Triggering Mechanism |
|---|:---:|:---:|:---:|---|
| `playback_issue` | 33 | **26** | **78.8%** | Scored $s \in [0.55, 0.72] < 0.73$ + 5 cross-domain misclassifications |
| `account_login` | 13 | **8** | **61.5%** | Standard password/login issues scoring $s \approx 0.65$ against SOP |
| `device_platform` | 11 | **7** | **63.6%** | Hardware-specific complaints failing dense match against generic mobile threads |
| `app_technical` | 9 | **6** | **66.7%** | Cache/crashing queries with elliptical phrasing below $\tau=0.73$ |
| `other` (Out-of-Tax) | 6 | **6** | **100.0%** | Unclassifiable chatter correctly failing high-similarity threshold |
| `premium_billing` | 11 | **5** | **45.5%** | Non-dispute billing questions (receipts, date checks) failing $0.73$ cutoff |

### Root Causes:
1. **Diverse Playback Vocabularies:** 26 of the 58 FEs (44.8%) concentrate in `playback_issue`. Terse customer queries ("shuffle looping", "songs keep pausing") yield top-1 cosine similarity scores between $0.58$ and $0.71$, falling just beneath the $\tau=0.73$ cutoff.
2. **Upstream Classification Bleed:** 17 of 58 FEs (29.3%) were misclassified by intent (e.g. 5 `playback_issue` cases classified as `app_technical`). Because retrieval is intent-scoped, this routed search to the wrong corpus, guaranteeing an artificially depressed similarity score.
3. **Layer 2 Dominance:** 57 of the 58 FEs were triggered by Layer 2 similarity falling below 0.73; only 1 was triggered by a Layer 1 hard gate (`billing_dispute` on an informational receipt check).

### 7.2 Phase 5: LLM-as-a-Judge Validation & Human Disagreement Analysis ($N=50$)

An independent, cross-architecture judge (`openai/gpt-oss-120b`, 120B reasoning model) was deployed over 50 stratified golden rows (all 35 auto-handled drafts + 15 representative escalations; 20 easy, 30 hard) to audit groundedness and tone without same-model bias:

| Difficulty Tier | Sample Size ($N$) | Raw Agreement | Cohen's Kappa ($\kappa$) | Asymptotic Std Error ($SE$) | 95% Confidence Interval |
|---|:---:|:---:|:---:|:---:|:---:|
| **Easy Tier** | 20 | 80.00% (16/20) | **0.375** | 0.280 | `[-0.173, 0.923]` |
| **Hard Tier** | 30 | 80.00% (24/30) | **0.392** | 0.222 | `[-0.043, 0.827]` |
| **Overall Dataset** | **50** | **80.00% (40/50)** | **0.381** | **0.175** | **`[0.038, 0.724]`** |

- **Continuous Metrics:** Groundedness MAE = **0.760** (Spearman $\rho = 0.313$), Tone MAE = **0.280** (Spearman $\rho = 0.076$).
- **Core Judge Failure Patterns:**
  1. *Procedure Sycophancy / Context Mismatch (Judge Passes, Human Fails — 3 cases):* The judge verifies that drafted advice literally matches SOP text, but ignores whether that SOP solves the customer's symptom. For example, in `gs_0004` and `gs_0116`, the customer complained about the shuffle algorithm's repetition; the agent drafted standard device restart steps (SOP PB_01). The human failed this as an unhelpful brush-off (G=2), while the judge passed it (G=5) because the steps strictly matched the SOP.
  2. *Hyper-Pedantic Narrowness (Judge Fails, Human Passes — 7 cases):* In cases like `gs_0007` (*Settings > Playback > Show unplayable songs*) and `gs_0085` (*Web account overview country edit*), the agent provided accurate Spotify menu paths that were omitted from the condensed SOP prompt. The human passed these as accurate troubleshooting (G=5), while the judge failed them (G=3) for citing setting names absent from the prompt snippet.
- **Synthesized Shared Mechanism:** Both *procedure sycophancy* and *evidence-narrowness* share the exact same root cause: the judge anchors on literal textual matching between the drafted reply and the SOP snippet, rather than reasoning about whether the underlying advice actually resolves the customer's stated problem. That literal anchoring produces leniency when the SOP is wrong-but-matched (passing unhelpful advice for shuffle algorithm or OS bugs), and hyper-strictness when the reply is right-but-unmatched (failing accurate settings paths omitted from the condensed SOP prompt).

---

## 8. What Is Misleading About My Headline Numbers

A core requirement of this project is explicitly confronting the ways headline metrics can flatter a system:

1. **Threshold Calibration Leakage:**
   Reporting $54.97\%$ accuracy and $14.71\%$ FAH on the full 151 rows introduces threshold snooping risk, as $\tau=0.73$ was tuned against that distribution. On the stratified held-out evaluation split ($N=76$), **FAH degraded to 20.00%**, failing the 15% hurdle.
2. **The Headline FAH Illusion (The Automation Collapse):**
   Claiming a "63% reduction in False Auto-Handles (from 39.7% to 14.7%)" sounds like a major safety breakthrough. But that number is deeply misleading without its operational counterpart: **False Escalations surged to 69.88%**, routing 70% of routine customer issues to human agents. The system achieved safety through defensive over-escalation rather than precise discrimination.
3. **The Groundedness Reality Gap (22% vs. 72%):**
   Heuristic sampling suggested 72% visible resolution. Rigorous hand-auditing proved that only **22.0%** of historical threads contained standalone public resolutions. Presenting drafted replies as "grounded in historical support cases" is misleading for 78% of the traffic, which actually relies on synthesized policy fallbacks.
4. **Retrieval Spot-Check Metric Inflation (30% vs. 90%):**
   An initial spot-check categorized 18/20 (90%) of retrieval hits as "relevant." Closer inspection revealed that 12 of those 18 were merely topically adjacent inquiries ending in DM redirects or diagnostic questions. Only **30% (6/20)** provided a concrete actionable resolution. Conflating topical relevance with operational resolution is a classic evaluation trap.
5. **Validation-Set Reuse & Small-N Asymptotic Variance:**
   All 50 human-judge validation rows were drawn from the same 151-row golden set already used to curate labels, calibrate $\tau=0.73$, and report headline agent metrics. Consequently, the 80.00% agreement ($\kappa = 0.381$) measures fit to an already-exposed label distribution, not generalization to unseen production queries. Furthermore, with $N_{easy}=20$ and $N_{hard}=30$, asymptotic standard errors are large ($SE \approx 0.22–0.28$), meaning the true agreement bounds span a wide range ($95\% \text{ CI}: [-0.04, 0.83]$). Point estimates alone significantly overstate statistical precision.

---

## 9. Next Steps (Phase 6 Finalization & Future Enhancements)

- **Phase 5 (Completed):** Automated evaluation harness (`eval/metrics.py`), independent cross-architecture LLM judge (`eval/judge.py`), and disaggregated human validation with failure pattern discovery (`eval/validate_judge.py`) are fully implemented and verified.
- **Phase 6 (Immediate Deliverable):**
  - Synthesize end-to-end findings across all phases into final submission deliverables.
  - Finalize the top-5 concrete failure mode taxonomy with real tweet pairs.
  - Compile final decision log entries into submission format.
- **Future Architectural Next Steps (Post-Evaluation Cycle):**
  - **Judge Rubric Redesign (Two-Stage Decoupling):** Redesign the judge rubric into a two-stage evaluation: first assess whether the customer's stated problem is meaningfully addressed, independent of SOP-text match, before scoring evidence groundedness. This directly resolves the literal textual match failure modes discovered in Phase 5, and was deliberately deferred to maintain the frozen-prompt anti-tuning commitment for the current evaluation cycle.


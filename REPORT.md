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

Every engineering omission was an intentional architectural tradeoff informed by empirical data:

1. **Multi-Lingual Processing (English-Only Restriction):** Screened out non-English traffic (~4%) to focus strictly on semantic nuance and policy adherence in the primary language (Decision Log Entry 5).
2. **Full TWCS Ingestion & Real-Time Streaming:** Ingesting 2.8M rows in real-time creates a severe memory and latency bottleneck. Scoped the operational pipeline to an 8,310-document visible-resolution index and a 151-row golden set, guaranteeing cold-start reproduction under 3.5 minutes (Decision Log Entry 3, Entry 6).
3. **Weight Fine-Tuning:** Avoided opaque gradient updates; prioritized few-shot in-context learning and dense retrieval for live auditability and rapid policy iteration (Decision Log Entry 8).
4. **Synthetic URL Generation:** Stripped all synthesized external support URLs from SOPs and banned link generation in drafter and judge prompts to eliminate hallucination risk (Decision Log Entry 17).
5. **Direct Tweet-to-Tweet Grounding for All Traffic:** Abandoned pure nearest-neighbor historical tweet reply copying after proving that 78.0% of historical brand replies lack standalone public resolutions. Built `CuratedPolicyReference` (17 SOPs) as a deterministic fallback (Decision Log Entry 16).
6. **Multi-Partition Cross-Encoder Re-Ranking:** Scoped dense retrieval strictly to the predicted intent partition to keep per-turn latency under 50ms, accepting the 29.3% upstream classification bleed failure mode as an operational tradeoff (Decision Log Entry 10).
7. **Same-Model Judge Self-Evaluation:** Refused to allow `qwen/qwen3.8-27b` to evaluate its own drafts; provisioned `openai/gpt-oss-120b` as an independent cross-architecture judge (Decision Log Entry 20).
8. **Circular Prompt Tuning on the Judge:** Committed to freezing the judge prompt after design rather than iteratively tuning it to artificially flatter human agreement (Decision Log Entry 21).

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

## 7. Comprehensive Failure Analysis & Error Taxonomy

A dependable engineering report must confront failure systematically. Auditing predictions across the 151 golden rows and 50 judge-audited cases isolates five recurring failure modes that govern system performance:

### 7.1 The Top 5 Concrete Failure Modes (with Real Tweet Audits)

#### Failure Mode 1: Upstream Intent Classification Bleed into Partitioned Retrieval Isolation
- **Error Category:** Cascading Pipeline Error (Intent Classifier $\rightarrow$ Scoped Retrieval).
- **Prevalence:** 17 of 58 False Escalations (**29.3%**); predominantly `playback_issue` misclassified as `app_technical`.
- **Root Mechanism:** Dense retrieval is strictly partitioned by predicted intent to eliminate cross-domain topic pollution. When a customer describes playback freezing using terms like *"app freezes"* or *"crashes"*, the few-shot classifier misclassifies the query into `app_technical`. Scoped retrieval then searches only the app-crash corpus, failing to find audio buffer SOPs. Cosine similarity plunges to $s \in [0.46, 0.66] < 0.73$, forcing defensive False Escalation.
- **Verbatim Evidence:**
  - *Case `gs_0000`:* Customer asks: *"@115888 why you don’t have a block song or artist thingy? N my feed this certain song keeps being played in my mixes & I can’t stop it 😡"*. Gold: `playback_issue`. Pred: `app_technical`. Scored $s = 0.467 < 0.73$. Routing: Falsely escalated due to search partition mismatch.
  - *Case `gs_0011`:* Customer asks: *"@SpotifyCares This is the 4th or 5th time I've noticed that this particular song is next in the queue when the app freezes."* Gold: `playback_issue`. Pred: `app_technical`. Scored $s = 0.612 < 0.73$. Falsely escalated because "freezes" triggered application crash partition.

#### Failure Mode 2: The Colloquial Phrasing Gap in the Dense Retrieval Dead Zone ($s \in [0.55, 0.72]$)
- **Error Category:** Calibrated Boundary Conservatism (Automation Forfeiture).
- **Prevalence:** **40 of 58 False Escalations (69.0%)**; accounts for 40 of the 57 Layer 2 similarity-triggered FEs, heavily concentrated in `playback_issue` (26/33 handleables, 78.8%).
- **Root Mechanism:** Terse, informal customer tweets describing routine audio glitches ("music stopped", "skipping", "pausing") lack the formal vocabulary of technical SOPs or past resolved cases. While the classifier correctly predicts `playback_issue`, dense cosine similarity scores cluster between $0.58$ and $0.71$. Under the safety-first threshold $\tau = 0.73$ (mandated to keep $\text{FAH} \le 15\%$), the agent refuses to auto-handle, surrendering over two-thirds of routine complaints to human queues.
- **Verbatim Evidence:**
  - *Case `gs_0006`:* Customer asks: *"@SpotifyCares I use a Samsung Galaxy. No matter what song I play, they all skip multiple times."* Gold: `playback_issue` (Auto-Handle). Pred: `playback_issue` (Escalate). Scored $s = 0.711$ (just $0.019$ below $\tau=0.73$). Routing: Falsely escalated despite being a routine cache refresh candidate.
  - *Case `gs_0008`:* Customer asks: *"@SpotifyCares No i don’t get a message. Like if i skip a song all the music stops - or if i pick a new album to shuffle it just won’t play anything"*. Gold: `playback_issue` (Auto-Handle). Scored $s = 0.684 < 0.73$. Routing: Falsely escalated to specialist queue.
- **Footnote on Layer 1 Gate Misfire (Case `gs_0133`):** The 58th false escalation was not triggered by Layer 2 similarity, but by a single Layer 1 hard gate misfire. In `gs_0133`, the customer tweeted: *"@SpotifyCares I cannot up date my credit card its still loading?????????????????"*. The phrase *"credit card"* triggered the deterministic `billing_dispute` gate on an in-app stuck web-form loading issue, escalating unconditionally and bypassing Layer 2 similarity entirely (accounting for 1.7%, 1/58 of FEs). Together, Failure Mode 1 (17 cases, 29.3%), Failure Mode 2 (40 cases, 69.0%), and the Layer 1 misfire (1 case, 1.7%) completely and mutually exclusively account for all 58 false escalations ($17 + 40 + 1 = 58$).

#### Failure Mode 3: Context-Blind Procedure Execution / Brush-Off Advice (Procedure Sycophancy)
- **Error Category:** Procedural Sycophancy / Semantic Mismatch (Safety Failure).
- **Prevalence:** 3 of 10 False Auto-Handles audited in Phase 5; approved by LLM judge but rejected by human auditors.
- **Root Mechanism:** The drafter retrieves standard SOP `PB_01_BASIC_REFRESH` (Session Refresh & Device Restart) and drafts a polite, fluent tweet instructing the user to log out and reboot their phone. While superficially grounded in the SOP, device reboot cannot solve Spotify's shuffle algorithm logic or OS-level hardware API regressions; it represents an unhelpful canned response that frustrates users.
- **Verbatim Evidence:**
  - *Case `gs_0004`:* Customer complains: *"@SpotifyCares No. Just work it out and make sure when I shuffle it doesn’t play the same album for 5 songs in a row 🤷‍♀️ that simple"*. Gold: `escalate`. Pred: `auto_handle` ($s = 0.766$). Draft: *"We hear you! To fix the shuffle issue, please log out, fully close the app, restart your device, and log back in..."*. Human: FAIL (G=2, unhelpful brush-off). Judge: PASS (G=5, literal SOP compliance).
  - *Case `gs_0111`:* Customer complains: *"@115888 @115858 our headphones have not control on Spotify app since ios 11 came. When will we have a fixed version??"*. Gold: `escalate` (OS API regression). Pred: `auto_handle` ($s = 0.758$). Draft: Recommends in-app Bluetooth re-pairing. Human: FAIL (G=3, hardware/OS API regression cannot be fixed by in-app re-pairing). Judge: PASS (G=5, literal SOP compliance).

#### Failure Mode 4: Sparse Complaint SOP Semantic Misattribution
- **Error Category:** Lexical Overfitting on Under-Specified Queries.
- **Prevalence:** Concentrated in `premium_billing` edge cases ($N=3$).
- **Root Mechanism:** When a customer tweets an ambiguous, terse billing inquiry ("Need help with subscription", "card declined", "where do I enter promo code?"), dense retrieval matches generic payment/billing tokens to `BILL_03_FAMILY_INVITATION` (Family & Duo Plan Address Match SOP). The agent drafts instructions about Family Plan manager invite links, sending bizarrely irrelevant advice to an individual subscriber.
- **Verbatim Evidence:**
  - *Case `gs_0127`:* Customer tweets: *"Just Cancel my Spotify because they didn’t let me buy @12387 tickets still luv @115888 I’m jus upset."* Draft: *"We’re sorry you’re upset. To resolve this, please have the plan manager generate a new invite link. The invitee must open it in a Private/Incognito window and enter the exact same residential address as the manager."* Card transaction failure resulted in Family Plan invite troubleshooting.
  - *Case `gs_0138`:* Customer asks: *"@SpotifyCares Hi, I cant find an Indosat option to buy spotify premium."* Draft: Instructs customer to enter residential home address matching the plan manager in an Incognito window. Missing carrier billing option resulted in Family Plan invite steps.

#### Failure Mode 5: Visible Resolution Deficit & Standalone Public Fallback Gap
- **Error Category:** Historical Corpus Resolution Deficit.
- **Prevalence:** Affects **78.0%** of historical visible-resolution threads; forces **94.3% of agent auto-handles (33/35)** to fall back to Curated SOPs (with only 2/35, 5.7%, grounded directly in historical visible resolutions, and 1.3% across the entire 151-row golden set).
- **Root Mechanism:** Rigorous auditing of historical threads in `twcs.csv` proved that 78.0% of brand replies are diagnostic questions ("What device/OS?") or DM deflections ("Please DM us your email /RI") rather than standalone resolutions. When dense retrieval returns non-actionable diagnostic questions, the agent is forced to fall back to the 17 generic SOPs. When a customer inquiry has subtle platform variations not captured in the 17 SOPs, the agent either forces a generic SOP (Modes 3 & 4) or falls below $\tau=0.73$ (Mode 2). *(This downstream drafting bottleneck directly produces the macro statistical gap detailed in Section 8, Item 3).*

---

### 7.2 Intent Breakdown of the 58 False Escalation Cases

Auditing the 58 handleable queries falsely routed to escalation at $\tau = 0.73$ highlights the operational price of safety:

| Intent Category | Handleable Rows ($N=83$) | Falsely Escalated (FE) | FE Rate within Intent | Primary Triggering Mechanism |
|---|:---:|:---:|:---:|---|
| `playback_issue` | 33 | **26** | **78.8%** | Scored $s \in [0.55, 0.72] < 0.73$ + 5 cross-domain misclassifications |
| `account_login` | 13 | **8** | **61.5%** | Standard password/login issues scoring $s \approx 0.65$ against SOP |
| `device_platform` | 11 | **7** | **63.6%** | Hardware-specific complaints failing dense match against generic mobile threads |
| `app_technical` | 9 | **6** | **66.7%** | Cache/crashing queries with elliptical phrasing below $\tau=0.73$ |
| `other` (Out-of-Tax) | 6 | **6** | **100.0%** | Unclassifiable chatter correctly failing high-similarity threshold |
| `premium_billing` | 11 | **5** | **45.5%** | Non-dispute billing questions (receipts, date checks) failing $0.73$ cutoff |

---

### 7.3 Phase 5: LLM-as-a-Judge Validation & Human Disagreement Analysis ($N=50$)

An independent, cross-architecture judge (`openai/gpt-oss-120b`, 120B reasoning model) was deployed over 50 stratified golden rows (all 35 auto-handled drafts + 15 representative escalations; 20 easy, 30 hard) to audit groundedness and tone without same-model bias:

| Difficulty Tier | Sample Size ($N$) | Raw Agreement | Cohen's Kappa ($\kappa$) | Asymptotic Std Error ($SE$) | 95% Confidence Interval |
|---|:---:|:---:|:---:|:---:|:---:|
| **Easy Tier** | 20 | 80.00% (16/20) | **0.375** | 0.280 | `[-0.173, 0.923]` |
| **Hard Tier** | 30 | 80.00% (24/30) | **0.392** | 0.222 | `[-0.043, 0.827]` |
| **Overall Dataset** | **50** | **80.00% (40/50)** | **0.381** | **0.175** | **`[0.038, 0.724]`** |

- **Continuous Metrics:** Groundedness MAE = **0.760** (Spearman $\rho = 0.313$), Tone MAE = **0.280** (Spearman $\rho = 0.076$).

### 7.4 The Shared Root Mechanism: Literal Anchoring vs. Problem Resolution
Crucially, *procedure sycophancy* (Judge PASS, Human FAIL — 3 cases) and *evidence-narrowness* (Judge FAIL, Human PASS — 7 cases) are not separate, unrelated flaws; they are two sides of the exact same underlying mechanism:
> **The judge anchors on literal textual matching between the drafted reply and the SOP snippet, rather than reasoning about whether the underlying advice actually resolves the customer's stated problem.**

- **Asymmetric Leniency (Wrong-but-Matched):** When an irrelevant procedure happens to match the retrieved SOP text verbatim (e.g. advising device restarts for shuffle algorithm complaints in `gs_0004` and `gs_0116`, or Bluetooth re-pairing for iOS 11 hardware volume control bugs in `gs_0111`), the judge awards a perfect 5/5 Groundedness and PASSES the reply, completely blind to customer symptom divergence.
- **Asymmetric Hyper-Strictness (Right-but-Unmatched):** When the drafted reply provides genuine, highly accurate domain troubleshooting (e.g. navigating to *Settings > Playback > Show unplayable songs* for local greyed-out tracks in `gs_0007`, or editing country settings on the web portal for overseas travel locks in `gs_0085`), the judge FAILS the reply because those exact UI paths were omitted from the condensed SOP prompt snippet.

---

## 8. What Is Misleading About My Headline Numbers

A core requirement of this project is explicitly confronting the ways headline metrics can flatter a system:

1. **Threshold Calibration Leakage:**
   Reporting $54.97\%$ accuracy and $14.71\%$ FAH on the full 151 rows introduces threshold snooping risk, as $\tau=0.73$ was tuned against that distribution. On the stratified held-out evaluation split ($N=76$), **FAH degraded to 20.00%**, failing the 15% hurdle.
2. **The Headline FAH Illusion (The Automation Collapse):**
   Claiming a "63% reduction in False Auto-Handles (from 39.7% to 14.7%)" sounds like a major safety breakthrough. But that number is deeply misleading without its operational counterpart: **False Escalations surged to 69.88%**, routing 70% of routine customer issues to human agents. The system achieved safety through defensive over-escalation rather than precise discrimination.
3. **The Groundedness Reality Gap (22% vs. 72%):**
   Heuristic sampling initially suggested a 72.0% visible-resolution rate. Rigorous hand-auditing proved that only **22.0%** of historical threads contained standalone public resolutions (see Section 7.1, Failure Mode 5 for concrete downstream tweet examples). Presenting drafted replies as "grounded in historical support cases" is misleading for 78% of the traffic, which actually relies on synthesized policy fallbacks.
4. **Retrieval Spot-Check Metric Inflation (30% vs. 90%):**
   An initial spot-check categorized 18/20 (90%) of retrieval hits as "relevant." Closer inspection revealed that 12 of those 18 were merely topically adjacent inquiries ending in DM redirects or diagnostic questions. Only **30% (6/20)** provided a concrete actionable resolution. Conflating topical relevance with operational resolution is a classic evaluation trap.
5. **Validation-Set Reuse & Small-N Asymptotic Variance:**
   All 50 human-judge validation rows were drawn from the same 151-row golden set already used to curate labels, calibrate $\tau=0.73$, and report headline agent metrics. Consequently, the 80.00% agreement ($\kappa = 0.381$) measures fit to an already-exposed label distribution, not generalization to unseen production queries. Furthermore, with $N_{easy}=20$ and $N_{hard}=30$, asymptotic standard errors are large ($SE \approx 0.22–0.28$), meaning the true agreement bounds span a wide range ($95\% \text{ CI}: [-0.04, 0.83]$). Point estimates alone significantly overstate statistical precision.

---

## 9. Next Steps (Future Enhancements & Reproducibility Guarantees)

### 9.1 Reproducibility Benchmarks
The codebase provides two distinct, explicitly separated verification modes:
1. **Verification Mode A (Instant Cached Replay, < 2 seconds):** Runs `python3 -m eval.metrics` and `python3 -m eval.validate_judge` to mathematically verify reported numbers against disk artifacts in ~1.5s.
2. **Verification Mode B (Cold-Start Full Regeneration, ~2.5 to 3.5 minutes):** Clean execution from scratch (regenerating embeddings across 8,310 docs in ~7.6s, live classification of 132 rows in ~45s, live escalation routing and drafting of 151 rows in ~45s, and live judge evaluation of 50 rows in ~40s), finishing comfortably under the 15-minute assignment limit.

### 9.2 Strategic Next Steps (Post-Evaluation Roadmap)
1. **Judge Rubric Redesign (Two-Stage Decoupling):** Redesign the judge rubric into a two-stage evaluation: first assess whether the customer's stated problem is meaningfully addressed, independent of SOP-text match, before scoring evidence groundedness. This directly resolves the literal textual match failure modes discovered in Phase 5, and was deliberately deferred to maintain the frozen-prompt anti-tuning commitment for the current evaluation cycle.
2. **Soft Multi-Partition Dense Retrieval:** Expand retrieval to search across the top-2 predicted intent corpora with cross-encoder re-ranking, eliminating the 29.3% of false escalations caused by upstream classification bleed.
3. **Query Expansion / Paraphrase Rewriting:** Normalize terse, colloquial customer tweets into standard technical symptom descriptions before computing dense embeddings, bridging the 70.7% colloidal phrasing dead zone.



# Phase 1b — Golden Eval Set: Methodology Note

## Purpose of this document

This note documents exactly how the 200-example golden eval set was constructed and
sampled — before any agent exists — so that a skeptical reviewer can assess whether
the eval is trustworthy or whether it contains leakage, selection bias, or measurement
artifacts. It is written immediately after sampling and before any labelling begins.

---

## Pool construction

**Source:** `twcs.csv` (Customer Support on Twitter, 2,811,774 rows total).

**Brand:** SpotifyCares only.

**Two-path pool construction** (both paths verified in Phase 1 data checks):

| Path | Description | Inbound tweets |
|---|---|---|
| Original filter | Tweets `in_response_to` a SpotifyCares tweet (customer replies mid-thread) | 15,096 |
| Alternate join | SpotifyCares outbound `in_response_to_tweet_id` → customer root tweet in dataset | 30,084 net new |
| **Combined** | | **45,180** |

The alternate join was verified with a 10-pair coherence spot-check before being
accepted. Thin tweets (customer text ≤ 30 chars after stripping @handle) were removed
(5,698 removed). After English filter (ASCII-ratio ≥ 85%): **39,379 usable inbound
tweets**.

> [!NOTE]
> The 30-char minimum removes entries like "Hi please see my DM :)" and "Android 7.0"
> — tweets where no classifiable intent is recoverable from the customer message alone.
> This is a quality gate, not a coverage cut.

---

## Stratified sampling plan

**Target total:** 150 examples (within the 150–250 cap, revised down from 200 via proportional trim to manage labelling fatigue risk while keeping the exact stratification shape intact).

### Intent allocation (150 total, ~70% easy / ~30% hard)

| Intent | Easy | Hard | Total | Rationale |
|---|---|---|---|---|
| `playback_issue` | 24 | 10 | 34 | Most frequent in data; failure-language filtered |
| `app_technical` | 21 | 9 | 30 | Distinct resolution path (reinstall/cache); equal weight |
| `account_login` | 21 | 9 | 30 | High visible-resolution rate; equal weight |
| `device_platform` | 21 | 9 | 30 | Platform-specific; equal weight |
| `premium_billing` | 18 | 8 | 26 | Slightly fewer because billing DM-redirect rate is higher (less visible resolution), and billing disputes trigger the hard escalation gate |
| **Total** | **105 (70.0%)** | **45 (30.0%)** | **150** | |

### Difficulty split

~30% "hard" per intent (~8–10 per intent), ~70% "easy". Hard examples were
identified by any of three heuristics:

1. Customer text ≤ 50 chars after stripping @handle (short = ambiguous)
2. Customer text matches keyword patterns for ≥ 2 intent buckets simultaneously
3. Customer text contains escalation-relevant keywords (hacked, fraud, charge, refund,
   legal, etc.) — these require an extra labelling decision (auto_handle vs escalate)

Actual difficulty labels (`easy`/`hard`) are assigned by the human labeller during
labelling, not by this heuristic. The heuristic `difficulty_hint` column in the CSV
is a **sampling guide only** — the labeller should override it freely when reading the
actual tweet.

---

## Keyword pre-bucketing (for sampling only)

Each inbound tweet was pre-assigned a `heuristic_bucket` by keyword match, used
purely to ensure enough candidates in each intent's sampling pool. The heuristic is
explicitly NOT the classifier and does NOT constrain the labeller.

**Heuristic bucket distribution (pre-sample pool, N=26,262 with a recognized bucket):**

| Bucket | Pool size | Notes |
|---|---|---|
| `premium_billing` | 7,629 | Large; many financial-keyword tweets |
| `playback_issue` | 6,922 | Large; broad keyword coverage |
| `account_login` | 4,026 | |
| `device_platform` | 3,688 | |
| `app_technical` | 3,121 | Smallest recognized bucket |
| `unclassified` | 13,993 | No keyword match — excluded from sampling |

### Known sampling biases — stated explicitly

1. **Unclassified exclusion bias.** 13,993 tweets matched no keyword and were not
   sampled. These are likely: very vague requests ("Help me"), meta-complaints about
   Spotify generally with no technical detail, and non-complaint @mentions. Excluding
   them means the golden set over-represents tweets with enough specificity to keyword-
   match, which is also the population where classification is easiest. Hard, vague
   complaints are underrepresented. This is disclosed, not hidden.

2. **Keyword-heuristic contamination in `playback_issue`.** The first sampled examples
   show feature requests ("add a random song button", "block song feature") matching
   `playback_issue` keywords. These are not playback support issues. The human labeller
   will catch these; they may end up labelled as `unclassified`/`other` or reassigned.
   This contamination is expected from a keyword heuristic and is why labelling is human,
   not automated.

3. **`premium_billing` hard-pool skew.** The billing bucket had 4,864 hard examples
   vs 2,973 easy — more hard than easy, which is unusual. Likely cause: billing tweets
   often contain escalation keywords (charge, refund) which trigger the hard heuristic.
   Only 25 easy + 10 hard were sampled (10 < normal 30% of 35 = 10.5, so essentially
   at cap). The labeller should be aware billing examples may skew harder than other
   intents.

4. **DM-redirect concentration in `account_login` and `premium_billing`.** Visible-
   resolution rate for sampled examples: account_login = 17/40 (43%), premium_billing
   = 17/35 (49%). For these intents, groundedness evaluation will be limited to the
   subset with visible resolutions, per the Phase 1 caveat in context.md.

---

## Labelling instructions

File: `golden_set_to_label.csv` (151 rows finalized; originally targeted 200, trimmed to 150, topped up 1 to satisfy floor)

**Columns to fill in (currently blank):**

| Column | Values | Instructions |
|---|---|---|
| `gold_intent` | `playback_issue`, `app_technical`, `account_login`, `device_platform`, `premium_billing` | Read the customer tweet. Assign the best-fitting intent from the committed taxonomy. If it genuinely fits none (feature request, general praise, pure @mention with no complaint), write `other` — but treat this as a failure of the sampling heuristic, not a new taxonomy category. |
| `has_visible_resolution` | `true` / `false` | Does the brand reply (`brand_reply_text`) contain actual troubleshooting steps or a concrete resolution? `false` if it's a DM-redirect, a bare link, or "we'll look into it." Use the pre-computed `has_brand_reply_visible_resolution` as a guide but override if wrong. |
| `gold_decision` | `auto_handle` / `escalate` | Would you trust this type of reply to be sent automatically, or does it need human review? **Default to `escalate` when unsure** — asymmetric error weighting per context.md. Hard gates: billing disputes → `escalate`; account security / hacked account → `escalate`; any legal language → `escalate`. |
| `gold_decision_reason` | short string | One-line reason: "billing dispute — hard gate", "simple playback fix, standard troubleshooting", "hacked account — security risk", etc. |
| `difficulty_tier` | `easy` / `hard` | Override the `difficulty_hint` column as you see fit. Hard = you had to think for more than a moment about which intent it belongs to, or the escalation decision wasn't obvious. |

**Labelling discipline:**
- Label each tweet from the customer text alone first. Then look at `brand_reply_text`
  only to fill `has_visible_resolution` and to sanity-check your intent label.
- Do not look at the `heuristic_bucket` column while labelling — it will bias you
  toward confirming the keyword assignment. Cover it or ignore it.
- If a tweet could plausibly fit two intents: assign the one you'd route it to in a
  real support system, then note the ambiguity in `gold_decision_reason`.
- Aim for consistent label quality throughout. If fatigue sets in after ~100 examples,
  stop and resume fresh — rushed back-half labels are the biggest risk to this eval set.

---

## Leakage prevention

- The 151 finalized examples are the **golden set only** — they are never used as
  few-shot training examples for the classifier or retrieval system (Phases 3–4).
- Training-pool examples are drawn from the remaining **39,228 unsampled threads**
  in the English pool (39,379 cleaned English inbound tweets − 151 golden rows).
- This disjointness is enforced by `tweet_id` / `customer_tweet_id`: any few-shot
  selection script explicitly excludes all IDs present in `golden_set_to_label.csv`.

---

## Evaluation Scoring Policy for `other` Rows (Option A — Locked Architectural Choice)

The intent taxonomy is strictly fixed at 5 intents per Phase 1 commitments (`playback_issue`, `app_technical`, `account_login`, `device_platform`, `premium_billing`). 19 of the 151 golden rows are gold-labelled `other` (out-of-taxonomy feature requests, wishlist ideas, and general feedback). Per Decision Log Entry 13:

- **Intent Classification Accuracy:** Evaluated exclusively on the **132 in-taxonomy rows** (denominator = 132: 52 playback, 27 account, 23 billing, 15 app_technical, 15 device_platform). The 5-intent classifier cannot predict `other` by definition; penalizing it for out-of-taxonomy inputs would distort per-intent calibration without testing intent disambiguation.
- **Escalation Decision Accuracy:** Evaluated on the **full 151 rows** (denominator = 151). Real deployment feeds inevitably contain noise and out-of-scope requests; what matters for system safety and trust is that the agent recognizes its own boundary and does not auto-handle unclassifiable requests with irrelevant canned technical troubleshooting. Safely escalating an out-of-taxonomy message is a true system success, while auto-handling it is a safety failure.

---

## Final Labelling Audit & Emergent Distribution (151 Examples)

Labelling of the golden evaluation set was completed 100% by human inspection across all 151 rows (zero synthetic LLM labels). Below is the final audited distribution across all dimensions:

### 1. Intent Breakdown
| Intent | Count | % of Golden Set | Notes / Routing Behaviour |
|---|---|---|---|
| `playback_issue` | 52 | 34.4% | Absorbed cross-device playback complaints & audio glitches |
| `account_login` | 27 | 17.9% | Strong retention, password/access/verification issues |
| `premium_billing` | 23 | 15.2% | High rate of mandatory escalation under hard gate |
| `other` | 19 | 12.6% | Feature requests, wishlist, out-of-scope non-malfunctions |
| `app_technical` | 15 | 9.9% | Met 15-row floor; crashes, blank screens, cache/reinstall |
| `device_platform` | 15 | 9.9% | Met 15-row floor via 1 clean top-up row (`gs_0150`) |
| **Total** | **151** | **100.0%** | |

### 2. Difficulty Distribution (Decision Log Entry 12)
- **Hard**: 101 rows (66.9%)
- **Easy**: 50 rows (33.1%)
*(Targeted ~70% easy / ~30% hard; emergent distribution was inverted due to real customer complexity. This provides substantially stronger validation power for Phase 5 LLM-as-judge evaluation).*

### 3. Visible In-Thread Resolution Rate (Groundedness Caveat)
- **Concrete visible resolution (`has_visible_resolution = true`)**: 33 rows (21.9%)
- **DM-redirect / Acknowledgment / Diagnostic-only (`has_visible_resolution = false`)**: 118 rows (78.1%)
*(Measured rate is 21.9% under the strict concrete fix definition, replacing the initial 72.0% heuristic estimate).*

### 4. Escalation Decision Distribution
- **`auto_handle`**: 83 rows (55.0%)
- **`escalate`**: 68 rows (45.0%)

---

## Acceptance check (per build.md Phase 1b)

- [x] Every committed intent has real coverage (minimum floor of 15 examples per intent achieved: playback=52, account=27, billing=23, tech=15, device=15, other=19)
- [x] Methodology note survives a skeptical read — fully documented with all biases, deviations, and contamination events logged in `context.md`
- [x] Labelling completed before Phase 3 (agent build) begins — 100% completed by human hand-labelling
- [x] Groundedness caveat updated with measured 22% rate vs 78% redirect/diagnostic rate
- [x] Difficulty inversion documented in Decision Log Entry 12


"""
src/baselines.py

Phase 2 — Baseline Establishment for Hiver AI Support Agent Evaluation.
Computes and evaluates all three required baselines against the 151-row golden set:
1. Trivial floor baseline (Always-escalate, Majority-class intent)
2. Classification-only baseline (TF-IDF + Logistic Regression on silver training pool)
3. Simple competitive baseline (Nearest historical resolution retrieval + rule-based escalation cutoff)

Strict leakage guard: The 151 golden set tweet IDs are strictly excluded from all training,
silver labelling, and retrieval index pools.
Evaluation policy: Per Decision Log Entry 13 (Option A), intent classification is evaluated
on the 132 in-taxonomy golden rows, and escalation decisions are evaluated on all 151 rows.
"""

import re
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = "/Users/kavya/Desktop/Groundcheck/archive/twcs/twcs.csv"
GOLDEN_PATH = "/Users/kavya/Desktop/Groundcheck/golden_set_to_label.csv"
BRAND = "SpotifyCares"
MIN_TEXT_LEN = 30
SEED = 42

# --- Keyword regexes for silver training labels ---
INTENT_PATS = {
    'device_platform': re.compile(
        r'\b(android|ios|iphone|ipad|mac|windows|pc|ps4|ps5|playstation|xbox|alexa|echo|tv|smart tv|chromecast|bluetooth|carplay|android auto|web player|browser)\b',
        re.IGNORECASE
    ),
    'app_technical': re.compile(
        r'\b(crash|freeze|won.?t open|not opening|won.?t load|not loading|error code|reinstall|uninstall|blank screen|black screen|glitch|cache)\b',
        re.IGNORECASE
    ),
    'account_login': re.compile(
        r'\b(login|log in|password|account|email|locked|access|forgot password|username|2fa|verification)\b',
        re.IGNORECASE
    ),
    'premium_billing': re.compile(
        r'\b(charge|charged|billing|payment|pay|refund|cancel|subscription|premium|fee|family plan)\b',
        re.IGNORECASE
    ),
    'playback_issue': re.compile(
        r'\b(play|playback|skip|skipping|shuffle|song|track|audio|sound|buffer|buffering|stream|pause|stop)\b',
        re.IGNORECASE
    ),
}

FEATURE_REQ_EXCLUDE_RE = re.compile(
    r'\b(add a|feature request|suggestion|would be cool|any plans to support|why is there not|why can.?t we|we need a|bring back)\b',
    re.IGNORECASE
)

# Hard escalation gates per system.md
ESCALATION_HARD_GATES_RE = re.compile(
    r'\b(charge|charged|billing|refund|cancel\s+subscription|payment|fee|invoice|bank|money|credit\s+card|'
    r'hacked|hack|unauthorized|stolen|compromised|breach|'
    r'lawsuit|attorney|lawyer|sue|suing|legal\s+action|police|harassment|threat)\b',
    re.IGNORECASE
)

DM_REDIRECT_RE = re.compile(
    r'\b(dm us|dm me|direct message|send us a (dm|message|private)|'
    r'private message|pm us|pm me|drop us a dm|send (us|me) a dm|please dm)\b',
    re.IGNORECASE
)

def strip_handle(text):
    return re.sub(r'^(@\w+\s*)+', '', str(text)).strip()

def detect_lang_fast(text):
    t = str(text).strip()
    if not t: return 'und'
    ratio = sum(1 for c in t if ord(c) < 128) / len(t)
    return 'en_likely' if ratio >= 0.85 else 'non_en'

def load_data():
    """Load twcs, perform join and filters, and strictly exclude golden set IDs."""
    print("Loading TWCS dataset...")
    df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    brand_out = df[df['author_id'] == BRAND].copy()
    brand_authored_ids = set(brand_out['tweet_id'].tolist())

    orig_mask = (df['author_id'] == BRAND) | df['in_response_to_tweet_id'].isin(brand_authored_ids)
    inbound_orig = df[orig_mask & (df['inbound'].astype(str).str.strip().str.lower() == 'true')].copy()

    brand_reply_targets = set(brand_out['in_response_to_tweet_id'].dropna().tolist())
    recoverable_ids = brand_reply_targets & set(df['tweet_id'].tolist())
    inbound_rec = df[df['tweet_id'].isin(recoverable_ids) & (df['inbound'].astype(str).str.strip().str.lower() == 'true')].copy()
    inbound_rec = inbound_rec[~inbound_rec['tweet_id'].isin(set(inbound_orig['tweet_id']))]

    all_inbound = pd.concat([inbound_orig, inbound_rec], ignore_index=True)
    all_inbound['clean'] = all_inbound['text'].apply(strip_handle)
    en = all_inbound[all_inbound['text'].apply(detect_lang_fast) == 'en_likely'].copy()
    en = en[en['clean'].str.len() >= MIN_TEXT_LEN].copy()

    golden_df = pd.read_csv(GOLDEN_PATH)
    golden_ids = set(golden_df['customer_tweet_id'].astype(str))

    # Strict leakage prevention
    pool = en[~en['tweet_id'].astype(str).isin(golden_ids)].copy()

    # Brand reply map
    brand_replies = {}
    for _, row in brand_out.iterrows():
        in_resp = row.get('in_response_to_tweet_id')
        if pd.notna(in_resp) and in_resp not in brand_replies:
            brand_replies[str(in_resp)] = str(row['text'])
    pool['brand_reply'] = pool['tweet_id'].astype(str).map(brand_replies)

    return pool, golden_df

def assign_silver_labels(pool):
    """Assign unambiguous single-intent silver labels to training pool."""
    def get_silver(text):
        if FEATURE_REQ_EXCLUDE_RE.search(text):
            return None
        matched = [k for k, p in INTENT_PATS.items() if p.search(text)]
        return matched[0] if len(matched) == 1 else None

    pool['silver'] = pool['clean'].apply(get_silver)
    return pool

def evaluate_baseline_1_trivial(golden_df):
    """Baseline 1: Trivial Floor."""
    print("\n========================================================")
    print("BASELINE 1: TRIVIAL FLOOR")
    print("========================================================")
    # Intent: Majority class on in-taxonomy rows (N=132)
    gold_in_tax = golden_df[golden_df['gold_intent'] != 'other'].copy()
    maj_intent = "playback_issue"
    maj_preds = [maj_intent] * len(gold_in_tax)
    maj_acc = accuracy_score(gold_in_tax['gold_intent'], maj_preds)
    print(f"1. Majority Class Intent Accuracy (in-tax, N={len(gold_in_tax)}): {maj_acc*100:.2f}%")

    # Escalation: Always escalate on full golden set (N=151)
    y_true = golden_df['gold_decision']
    y_pred = ["escalate"] * len(golden_df)
    esc_acc = accuracy_score(y_true, y_pred)
    fah = sum(1 for t, p in zip(y_true, y_pred) if t == "escalate" and p == "auto_handle")
    fe = sum(1 for t, p in zip(y_true, y_pred) if t == "auto_handle" and p == "escalate")
    print(f"2. Always-Escalate Decision Accuracy (N={len(golden_df)}): {esc_acc*100:.2f}%")
    print(f"   False Auto-Handles (FAH): {fah}/68 (0.0% missed escalations — maximally safe)")
    print(f"   False Escalations (FE): {fe}/83 (100.0% handleable queries escalated — zero automation)")

    return {
        'intent_acc': maj_acc,
        'esc_acc': esc_acc,
        'fah': fah,
        'fe': fe,
    }

def evaluate_baseline_2_classification_only(pool, golden_df):
    """Baseline 2: Classification-Only (TF-IDF + Logistic Regression)."""
    print("\n========================================================")
    print("BASELINE 2: CLASSIFICATION-ONLY (TF-IDF + Logistic Regression)")
    print("========================================================")
    # Sample balanced training set: up to 250 per intent
    silver_pool = pool[pool['silver'].notna()].copy()
    train_samples = []
    for intent, grp in silver_pool.groupby('silver'):
        train_samples.append(grp.sample(n=min(len(grp), 250), random_state=SEED))
    train_df = pd.concat(train_samples, ignore_index=True)
    print(f"Trained on {len(train_df)} balanced silver samples from unsampled pool.")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train = vec.fit_transform(train_df['clean'])
    y_train = train_df['silver']

    clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    clf.fit(X_train, y_train)

    # Evaluate on in-taxonomy golden set (Option A: N=132)
    gold_in_tax = golden_df[golden_df['gold_intent'] != 'other'].copy()
    X_gold = vec.transform(gold_in_tax['customer_text'].apply(strip_handle))
    y_gold_true = gold_in_tax['gold_intent']
    y_gold_pred = clf.predict(X_gold)

    acc = accuracy_score(y_gold_true, y_gold_pred)
    print(f"Overall In-Taxonomy Accuracy (N=132): {acc*100:.2f}%\n")
    report = classification_report(y_gold_true, y_gold_pred, digits=3)
    print(report)

    return {
        'accuracy': acc,
        'report': report,
        'classifier': clf,
        'vectorizer': vec,
    }

def evaluate_baseline_3_simple(pool, golden_df):
    """Baseline 3: Simple Competitive (Nearest Historical Resolution + Rule Gate Cutoff)."""
    print("\n========================================================")
    print("BASELINE 3: SIMPLE RETRIEVAL + RULE-BASED ESCALATION CUTOFF")
    print("========================================================")
    # Build retrieval index: training pool items with visible resolution & silver intent
    pool['has_vis_res'] = pool['brand_reply'].apply(
        lambda r: pd.notna(r) and len(str(r)) >= 80 and not bool(DM_REDIRECT_RE.search(str(r)))
    )
    ret_pool = pool[pool['has_vis_res'] & pool['silver'].notna()].copy()
    print(f"Retrieval index size: {len(ret_pool):,} historical visible-resolution threads.")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_ret = vec.fit_transform(ret_pool['clean'])

    X_gold = vec.transform(golden_df['customer_text'].apply(strip_handle))
    sims = cosine_similarity(X_gold, X_ret)

    top_idxs = sims.argmax(axis=1)
    top_scores = sims.max(axis=1)
    ret_intents = ret_pool['silver'].values[top_idxs]

    # Retrieval intent accuracy (N=132 in-tax)
    gold_in_tax_mask = golden_df['gold_intent'] != 'other'
    ret_intent_acc = accuracy_score(
        golden_df.loc[gold_in_tax_mask, 'gold_intent'],
        ret_intents[gold_in_tax_mask]
    )
    print(f"Top-1 Retrieval Intent Classification Accuracy (in-tax, N=132): {ret_intent_acc*100:.2f}%\n")

    # Escalation decision evaluation across calibrated thresholds
    y_true = golden_df['gold_decision']
    easy_mask = golden_df['difficulty_tier'] == 'easy'
    hard_mask = golden_df['difficulty_tier'] == 'hard'

    print("--- Escalation Decision Performance across Thresholds (N=151) ---")
    threshold_results = []
    for tau in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        preds = []
        for i, row in golden_df.iterrows():
            t = row['customer_text']
            s = top_scores[i]
            if ESCALATION_HARD_GATES_RE.search(t):
                preds.append('escalate')
            elif s >= tau:
                preds.append('auto_handle')
            else:
                preds.append('escalate')

        acc = accuracy_score(y_true, preds)
        acc_easy = accuracy_score(y_true[easy_mask], [preds[j] for j in range(len(preds)) if easy_mask.iloc[j]])
        acc_hard = accuracy_score(y_true[hard_mask], [preds[j] for j in range(len(preds)) if hard_mask.iloc[j]])
        fah = sum(1 for t, p in zip(y_true, preds) if t == 'escalate' and p == 'auto_handle')
        fe = sum(1 for t, p in zip(y_true, preds) if t == 'auto_handle' and p == 'escalate')

        threshold_results.append({
            'tau': tau, 'acc': acc, 'acc_easy': acc_easy, 'acc_hard': acc_hard,
            'fah': fah, 'fe': fe
        })
        print(f"tau={tau:.2f} | Overall: {acc*100:.1f}% | Easy: {acc_easy*100:.1f}% | Hard: {acc_hard*100:.1f}% | FAH: {fah:2d}/68 ({fah/68*100:4.1f}%) | FE: {fe:2d}/83 ({fe/83*100:4.1f}%)")

    return {
        'ret_intent_acc': ret_intent_acc,
        'thresholds': threshold_results,
    }

def main():
    pool, golden_df = load_data()
    pool = assign_silver_labels(pool)

    b1 = evaluate_baseline_1_trivial(golden_df)
    b2 = evaluate_baseline_2_classification_only(pool, golden_df)
    b3 = evaluate_baseline_3_simple(pool, golden_df)

    print("\n========================================================")
    print("PHASE 2 BASELINE SUMMARY COMPLETED")
    print("========================================================")

if __name__ == "__main__":
    main()

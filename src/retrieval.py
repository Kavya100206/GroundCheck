"""
src/retrieval.py

Phase 3 — Grounding / Retrieval Component for SpotifyCares.
Uses SentenceTransformers ('all-MiniLM-L6-v2') dense embeddings over the brand's
visible-resolution subset from the unsampled training pool (8,310 threads).

Strict leakage guard: The 151 golden set tweet IDs are strictly excluded from the corpus.
Supports both intent-scoped and global semantic similarity search.
Includes an automated 20-example manual spot-check audit suite.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import re
import socket
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = "/Users/kavya/Desktop/Groundcheck/archive/twcs/twcs.csv"
GOLDEN_PATH = "/Users/kavya/Desktop/Groundcheck/golden_set_to_label.csv"
CORPUS_CACHE_PATH = "/Users/kavya/Desktop/Groundcheck/src/resolution_corpus.csv"
EMBEDDINGS_CACHE_PATH = "/Users/kavya/Desktop/Groundcheck/src/resolution_embeddings.npy"
BRAND = "SpotifyCares"
MIN_TEXT_LEN = 30
MODEL_NAME = "all-MiniLM-L6-v2"

DM_REDIRECT_RE = re.compile(
    r'\b(dm us|dm me|direct message|send us a (dm|message|private)|'
    r'private message|pm us|pm me|drop us a dm|send (us|me) a dm|please dm)\b',
    re.IGNORECASE
)

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

def strip_handle(text):
    return re.sub(r'^(@\w+\s*)+', '', str(text)).strip()

def detect_lang_fast(text):
    t = str(text).strip()
    if not t: return 'und'
    ratio = sum(1 for c in t if ord(c) < 128) / len(t)
    return 'en_likely' if ratio >= 0.85 else 'non_en'

class GroundingRetriever:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.corpus_df = None
        self.embeddings = None
        self._initialize_corpus()

    def _initialize_corpus(self):
        """Build or load cached visible resolution corpus and embeddings."""
        if os.path.exists(CORPUS_CACHE_PATH) and os.path.exists(EMBEDDINGS_CACHE_PATH):
            print("Loading cached resolution corpus and embeddings...")
            self.corpus_df = pd.read_csv(CORPUS_CACHE_PATH)
            self.embeddings = np.load(EMBEDDINGS_CACHE_PATH)
            print(f"Loaded {len(self.corpus_df):,} cached resolution threads (embedding shape: {self.embeddings.shape}).")
            return

        print("Extracting visible-resolution historical corpus from TWCS...")
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
        all_inbound['clean_text'] = all_inbound['text'].apply(strip_handle)
        en = all_inbound[all_inbound['text'].apply(detect_lang_fast) == 'en_likely'].copy()
        en = en[en['clean_text'].str.len() >= MIN_TEXT_LEN].copy()

        # Strict leakage guard: Exclude all golden set IDs
        golden_df = pd.read_csv(GOLDEN_PATH)
        golden_ids = set(golden_df['customer_tweet_id'].astype(str))
        pool = en[~en['tweet_id'].astype(str).isin(golden_ids)].copy()

        # Join brand replies
        brand_replies = {}
        for _, row in brand_out.iterrows():
            in_resp = row.get('in_response_to_tweet_id')
            if pd.notna(in_resp) and in_resp not in brand_replies:
                brand_replies[str(in_resp)] = str(row['text'])
        pool['brand_reply_text'] = pool['tweet_id'].astype(str).map(brand_replies)

        # Assign silver intent for intent-scoped filtering (unambiguous 5-intent match)
        def get_silver(text):
            if FEATURE_REQ_EXCLUDE_RE.search(text): return None
            matched = [k for k, p in INTENT_PATS.items() if p.search(text)]
            return matched[0] if len(matched) == 1 else None

        pool['silver_intent'] = pool['clean_text'].apply(get_silver)

        # Filter strictly for visible resolutions: length >= 80, no DM redirect phrase
        pool['has_visible_resolution'] = pool['brand_reply_text'].apply(
            lambda r: pd.notna(r) and len(str(r)) >= 80 and not bool(DM_REDIRECT_RE.search(str(r)))
        )

        # Strict visible resolution index matching Baseline 3 (8,310 documents = ~21.2% of pool)
        vis_pool = pool[pool['has_visible_resolution'] & pool['silver_intent'].notna()].copy()

        # Select columns to persist and sanitize internal newlines to prevent CSV reload fragmentation
        cols = ['tweet_id', 'clean_text', 'brand_reply_text', 'silver_intent']
        vis_pool['clean_text'] = vis_pool['clean_text'].astype(str).str.replace('\r', ' ').str.replace('\n', ' ')
        vis_pool['brand_reply_text'] = vis_pool['brand_reply_text'].astype(str).str.replace('\r', ' ').str.replace('\n', ' ')
        self.corpus_df = vis_pool[cols].reset_index(drop=True)
        print(f"Computed strict visible resolution corpus: {len(self.corpus_df):,} threads (~{len(self.corpus_df)/len(pool)*100:.1f}% of pool).")

        # Compute embeddings
        print(f"Encoding {len(self.corpus_df):,} customer complaint queries with {MODEL_NAME}...")
        texts = self.corpus_df['clean_text'].tolist()
        self.embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

        # Save caches
        self.corpus_df.to_csv(CORPUS_CACHE_PATH, index=False)
        np.save(EMBEDDINGS_CACHE_PATH, self.embeddings)
        print(f"Cached resolution corpus and embeddings to disk.")

    def retrieve(self, query: str, intent: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar historical resolutions."""
        clean_q = strip_handle(query)
        q_emb = self.model.encode([clean_q], normalize_embeddings=True)[0]

        if intent and intent in INTENT_PATS:
            # Intent-scoped retrieval
            mask = (self.corpus_df['silver_intent'] == intent).values
            if mask.sum() >= top_k:
                sub_indices = np.where(mask)[0]
                sub_embs = self.embeddings[sub_indices]
                sims = np.dot(sub_embs, q_emb)
                top_local_idxs = np.argsort(sims)[::-1][:top_k]
                top_global_idxs = sub_indices[top_local_idxs]
                top_scores = sims[top_local_idxs]
            else:
                # Fall back to global if intent subset too small
                sims = np.dot(self.embeddings, q_emb)
                top_global_idxs = np.argsort(sims)[::-1][:top_k]
                top_scores = sims[top_global_idxs]
        else:
            # Global retrieval
            sims = np.dot(self.embeddings, q_emb)
            top_global_idxs = np.argsort(sims)[::-1][:top_k]
            top_scores = sims[top_global_idxs]

        results = []
        for idx, score in zip(top_global_idxs, top_scores):
            row = self.corpus_df.iloc[idx]
            results.append({
                "tweet_id": row["tweet_id"],
                "customer_text": row["clean_text"],
                "brand_reply": row["brand_reply_text"],
                "silver_intent": row["silver_intent"],
                "similarity": float(score)
            })
        return results

def run_spot_checks(retriever: GroundingRetriever, n_checks: int = 20):
    """Run manual inspection on 20 representative queries across all intents."""
    print("\n========================================================")
    print("PHASE 3 RETRIEVAL QUALITY SPOT-CHECK (20 QUERIES)")
    print("========================================================")
    golden_df = pd.read_csv(GOLDEN_PATH)
    sample_df = golden_df[golden_df['gold_intent'] != 'other'].sample(n_checks, random_state=42)

    spot_results = []
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        q = row['customer_text']
        intent = row['gold_intent']
        hits = retriever.retrieve(q, intent=intent, top_k=1)
        top_hit = hits[0] if hits else None
        
        spot_results.append({
            "id": row["id"],
            "gold_intent": intent,
            "difficulty": row["difficulty_tier"],
            "query": strip_handle(q),
            "retrieved_query": top_hit["customer_text"] if top_hit else "None",
            "retrieved_reply": top_hit["brand_reply"] if top_hit else "None",
            "similarity": top_hit["similarity"] if top_hit else 0.0
        })

    for i, res in enumerate(spot_results):
        print(f"\n--- Spot-Check #{i+1:02d} [{res['id']}] Intent: {res['gold_intent']} (Difficulty: {res['difficulty']}) ---")
        print(f"Customer Query:    {res['query'][:100]}...")
        print(f"Retrieved Match:   {res['retrieved_query'][:100]}... (Cosine Sim: {res['similarity']:.3f})")
        print(f"Historical Reply:  {res['retrieved_reply'][:120]}...")

    return spot_results

if __name__ == "__main__":
    retriever = GroundingRetriever()
    run_spot_checks(retriever, n_checks=20)

"""
src/agent.py

Phase 4 Complete Agent Pipeline for SpotifyCares.
Integrates:
1. Ingest: Customer text (+ thread context if multi-turn)
2. Classify: Few-shot intent classification via 'qwen/qwen3.8-27b' (src/classifier.py)
3. Retrieve: Dense semantic grounding via 'all-MiniLM-L6-v2' (src/retrieval.py)
4. Policy: Curated SOP fallback for non-concrete retrieval (src/policy.py)
5. Escalate: Two-layer deterministic gate + calibrated confidence (src/escalation.py)
6. Draft: Grounded Twitter reply generation via 'qwen/qwen3.8-27b' (src/drafter.py)

Outputs full schema per system.md and caches predictions to disk.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.classifier import IntentClassifier
from src.retrieval import GroundingRetriever, strip_handle
from src.policy import CuratedPolicyReference
from src.escalation import decide_escalation, DEFAULT_TAU
from src.drafter import ReplyDrafter

GOLDEN_PATH = "/Users/kavya/Desktop/Groundcheck/golden_set_to_label.csv"
AGENT_PREDICTIONS_CACHE = "/Users/kavya/Desktop/Groundcheck/src/agent_predictions_golden151.csv"

class SpotifySupportAgent:
    def __init__(self, tau: float = DEFAULT_TAU):
        self.tau = tau
        print("Initializing SpotifySupportAgent pipeline...")
        self.classifier = IntentClassifier(model_name="qwen/qwen3.8-27b")
        self.retriever = GroundingRetriever()
        self.policy = CuratedPolicyReference()
        self.drafter = ReplyDrafter(model_name="qwen/qwen3.8-27b")
        print(f"Agent initialized successfully with calibrated threshold tau={self.tau:.2f}.")

    def process(
        self,
        customer_text: str,
        message_id: str = "demo",
        known_intent: Optional[str] = None,
        known_confidence: Optional[float] = None,
        cached_draft: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run complete single-message inference pipeline."""
        # Type safety & degenerate input guard
        clean_text = strip_handle(customer_text) if customer_text is not None else ""
        if not clean_text:
            return {
                "id": message_id,
                "predicted_intent": "other",
                "drafted_reply": "We’re here to help! Could you please provide more details about the issue you’re experiencing so our team can look into it?",
                "decision": "escalate",
                "decision_reason": "Customer message is empty or too short to classify; routing to specialist team",
                "grounding_source": "none",
                "evidence_used": ["Degenerate input: Empty or handle-only message"],
                "gate_triggered": "degenerate_input",
                "calibrated_confidence": 0.0
            }

        # 1. Intent Classification (reuse cached result if provided)
        if known_intent is not None:
            predicted_intent = known_intent
            confidence = known_confidence if known_confidence is not None else 0.85
        else:
            clf_result = self.classifier.classify(customer_text)
            predicted_intent = clf_result.get("intent", "playback_issue")
            confidence = clf_result.get("confidence", 0.8)

        # 2. Dense Semantic Retrieval
        ret_hits = self.retriever.retrieve(customer_text, intent=predicted_intent, top_k=1)
        top_sim = ret_hits[0]["similarity"] if ret_hits else 0.0

        # 3. Two-Layer Escalation Decision
        esc_result = decide_escalation(
            customer_text=customer_text,
            predicted_intent=predicted_intent,
            retrieval_similarity=top_sim,
            classifier_confidence=confidence,
            tau=self.tau
        )
        decision = esc_result["decision"]
        decision_reason = esc_result["decision_reason"]
        gate_triggered = esc_result["gate_triggered"]

        # 4. Grounded Reply Drafting (reuse existing draft if decision did not change)
        if cached_draft is not None and cached_draft.get("decision") == decision and pd.notna(cached_draft.get("drafted_reply")):
            draft_result = {
                "drafted_reply": cached_draft["drafted_reply"],
                "grounding_source": cached_draft.get("grounding_source", "curated_sop"),
                "evidence_used": [cached_draft["evidence_used"]] if isinstance(cached_draft.get("evidence_used"), str) else cached_draft.get("evidence_used", [])
            }
        else:
            draft_result = self.drafter.draft_reply(
                customer_text=customer_text,
                predicted_intent=predicted_intent,
                decision=decision,
                decision_reason=decision_reason,
                retrieved_hits=ret_hits
            )

        # 5. Output Record adhering strictly to system.md schema
        return {
            "id": message_id,
            "predicted_intent": predicted_intent,
            "drafted_reply": draft_result["drafted_reply"],
            "decision": decision,
            "decision_reason": decision_reason,
            "grounding_source": draft_result["grounding_source"],
            "evidence_used": draft_result["evidence_used"],
            "gate_triggered": gate_triggered,
            "calibrated_confidence": esc_result["calibrated_confidence"]
        }

def evaluate_agent(tau: float = DEFAULT_TAU, use_cache: bool = True):
    """Run full agent evaluation over the 151 golden set rows with incremental persistence."""
    print(f"\n========================================================")
    print(f"PHASE 4 AGENT EVALUATION (N=151 Golden Set, tau={tau:.2f})")
    print(f"========================================================")
    golden_df = pd.read_csv(GOLDEN_PATH)
    y_true = golden_df["gold_decision"]
    easy_mask = golden_df["difficulty_tier"] == "easy"
    hard_mask = golden_df["difficulty_tier"] == "hard"

    # Load cached classifier predictions to avoid redundant LLM calls
    cached_clf = {}
    clf_cache_path = "/Users/kavya/Desktop/Groundcheck/src/classifier_predictions.csv"
    if os.path.exists(clf_cache_path):
        c_df = pd.read_csv(clf_cache_path)
        for _, r in c_df.iterrows():
            cached_clf[str(r["id"])] = (str(r["predicted_intent"]), float(r.get("confidence", 0.85)))
        print(f"Loaded {len(cached_clf)} pre-computed classifications from {clf_cache_path}")

    # Check for existing partial or complete agent runs
    existing_records = {}
    if os.path.exists(AGENT_PREDICTIONS_CACHE):
        try:
            cached_df = pd.read_csv(AGENT_PREDICTIONS_CACHE)
            for _, r in cached_df.iterrows():
                existing_records[str(r["id"])] = r.to_dict()
            print(f"Loaded {len(existing_records)} prior agent predictions from disk for smart draft reuse.")
        except Exception:
            existing_records = {}

    agent = SpotifySupportAgent(tau=tau)
    print(f"Executing agent inference over {len(golden_df)} golden set threads (tau={tau:.2f})...")
    start_time = time.time()
    records = []

    for idx, (_, row) in enumerate(golden_df.iterrows()):
        msg_id = str(row["id"])
        known_intent, known_conf = cached_clf.get(msg_id, (None, None))
        cached_d = existing_records.get(msg_id, None)

        out = agent.process(
            row["customer_text"],
            message_id=msg_id,
            known_intent=known_intent,
            known_confidence=known_conf,
            cached_draft=cached_d
        )
        out["difficulty_tier"] = row["difficulty_tier"]
        out["gold_decision"] = row["gold_decision"]
        out["gold_intent"] = row["gold_intent"]
        records.append(out)

        # Incremental save every 10 rows
        if (idx + 1) % 25 == 0 or (idx + 1) == len(golden_df):
            temp_df = pd.DataFrame(records)
            temp_df.to_csv(AGENT_PREDICTIONS_CACHE, index=False)
            print(f"  Processed {idx + 1}/{len(golden_df)} rows ({time.time() - start_time:.1f}s)...")

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(AGENT_PREDICTIONS_CACHE, index=False)
    print(f"Saved full agent predictions to {AGENT_PREDICTIONS_CACHE} ({time.time() - start_time:.1f}s).")

    y_pred = pred_df["decision"]
    acc_overall = accuracy_score(y_true, y_pred)
    acc_easy = accuracy_score(y_true[easy_mask], y_pred[easy_mask])
    acc_hard = accuracy_score(y_true[hard_mask], y_pred[hard_mask])

    fah = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "escalate" and yp == "auto_handle")
    fe = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "auto_handle" and yp == "escalate")

    print("\n--------------------------------------------------------")
    print("ESCALATION DECISION PERFORMANCE VS BASELINE 3 (tau=0.35)")
    print("--------------------------------------------------------")
    print(f"Overall Escalation Accuracy: {acc_overall*100:.2f}% (Baseline 3: 53.00%) -> Margin: {acc_overall*100 - 53.0:+.2f}%")
    print(f"  Easy Tier Accuracy (N={easy_mask.sum()}):  {acc_easy*100:.2f}% (Baseline 3: 56.00%)")
    print(f"  Hard Tier Accuracy (N={hard_mask.sum()}):  {acc_hard*100:.2f}% (Baseline 3: 51.50%)")
    print(f"False Auto-Handles (FAH):    {fah:2d}/68 ({fah/68*100:4.1f}%) (Baseline 3: 27/68, 39.7%)")
    print(f"False Escalations (FE):      {fe:2d}/83 ({fe/83*100:4.1f}%) (Baseline 3: 44/83, 53.0%)")

    print("\n--------------------------------------------------------")
    print("GROUNDING SOURCE BREAKDOWN")
    print("--------------------------------------------------------")
    print(pred_df["grounding_source"].value_counts())

    print("\n--------------------------------------------------------")
    print("GATE TRIGGER DISTRIBUTION")
    print("--------------------------------------------------------")
    print(pred_df["gate_triggered"].value_counts(dropna=False))

    print("\nSample Agent Outputs:")
    for _, r in pred_df.head(5).iterrows():
        print(f"\n[{r['id']}] Decision: {r['decision']} (Grounding: {r['grounding_source']}, Gate: {r['gate_triggered']})")
        print(f"  Draft:  {r['drafted_reply']}")
        print(f"  Reason: {r['decision_reason']}")

    return {
        "acc_overall": acc_overall,
        "acc_easy": acc_easy,
        "acc_hard": acc_hard,
        "fah": fah,
        "fe": fe,
        "pred_df": pred_df
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Calibrated confidence threshold")
    parser.add_argument("--no-cache", action="store_true", help="Force re-inference without cache")
    args = parser.parse_args()

    evaluate_agent(tau=args.tau, use_cache=not args.no_cache)

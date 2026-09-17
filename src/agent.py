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

import re
import time
import json
import argparse
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.classifier import IntentClassifier
from src.retrieval import GroundingRetriever, strip_handle
from src.policy import CuratedPolicyReference
from src.escalation import decide_escalation, check_hard_gates, DEFAULT_TAU
from src.resolution_check import ProblemResolutionChecker
from src.drafter import ReplyDrafter

GOLDEN_PATH = "/Users/kavya/Desktop/Groundcheck/golden_set_to_label.csv"
CLASSIFIER_PREDICTIONS_PATH = "/Users/kavya/Desktop/Groundcheck/src/classifier_predictions.csv"
AGENT_PREDICTIONS_CACHE = "/Users/kavya/Desktop/Groundcheck/src/agent_predictions_golden151.csv"

class SpotifySupportAgent:
    def __init__(
        self,
        tau: float = DEFAULT_TAU,
        tau_low: Optional[float] = 0.65,
        tau_high: Optional[float] = 0.73,
        retrieval_policy: str = "scoped",
        use_resolution_check: bool = True
    ):
        self.tau = tau
        self.tau_low = tau if not use_resolution_check else tau_low
        self.tau_high = tau if not use_resolution_check else tau_high
        self.retrieval_policy = retrieval_policy
        self.use_resolution_check = use_resolution_check
        print("Initializing SpotifySupportAgent pipeline...")
        self.classifier = IntentClassifier(model_name="qwen/qwen3.8-27b")
        self.retriever = GroundingRetriever()
        self.policy = CuratedPolicyReference()
        self.drafter = ReplyDrafter(model_name="qwen/qwen3.8-27b")
        if self.use_resolution_check:
            self.resolution_checker = ProblemResolutionChecker(model_name="openai/gpt-oss-120b")
        else:
            self.resolution_checker = None
        print(f"Agent initialized successfully (policy={self.retrieval_policy}, tau_low={self.tau_low}, tau_high={self.tau_high}, check={self.use_resolution_check}).")

    def _get_candidate_procedure(self, hit: Optional[Dict[str, Any]], intent: str, text: str) -> str:
        """Extract candidate troubleshooting procedure from historical match or curated SOP fallback."""
        if hit and self.drafter._is_concrete_resolution(hit.get("brand_reply", "")):
            return hit["brand_reply"]
        sop = self.policy.find_matching_sop(intent, text)
        if sop:
            return f"{sop['name']}: " + " ".join(sop['steps'])
        return "General Troubleshooting: Restart your device and refresh network connection."

    def process(
        self,
        customer_text: str,
        message_id: str = "demo",
        known_intent: Optional[str] = None,
        known_confidence: Optional[float] = None,
        cached_draft: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute full agent pipeline on a single customer message.
        """
        # 0. Intercept empty, None, or degenerate handle-only inputs
        clean_text = customer_text.strip() if customer_text is not None else ""
        text_without_handles = re.sub(r"@\w+", "", clean_text).strip()
        if not clean_text or not text_without_handles or len(text_without_handles) < 3:
            return {
                "id": message_id,
                "predicted_intent": "playback_issue",
                "drafted_reply": "We’re looking into this issue and routing your case to our specialist team for assistance. Thank you for your patience.",
                "decision": "escalate",
                "decision_reason": "Customer message is empty or too short to classify",
                "grounding_source": "none",
                "evidence_used": ["Escalation Routing: Degenerate or empty customer input"],
                "gate_triggered": "degenerate_input",
                "calibrated_confidence": 0.0
            }

        # 1. Intent Classification
        if known_intent is not None:
            predicted_intent = known_intent
            confidence = known_confidence if known_confidence is not None else 1.0
        else:
            clf_res = self.classifier.classify(customer_text)
            predicted_intent = clf_res["intent"]
            confidence = clf_res["confidence"]

        # 2. Dense Semantic Retrieval
        if self.retrieval_policy == "global":
            ret_hits = self.retriever.retrieve(customer_text, intent=None, top_k=1)
        else:
            ret_hits = self.retriever.retrieve(customer_text, intent=predicted_intent, top_k=1)
        top_sim = ret_hits[0]["similarity"] if ret_hits else 0.0

        # 3. Pre-drafting Problem-Resolution Verification Gate
        resolution_check_result = None
        t_low = self.tau_low if self.tau_low is not None else self.tau
        hard_gate = check_hard_gates(customer_text, predicted_intent)
        if hard_gate is None and self.use_resolution_check and self.resolution_checker is not None and top_sim >= t_low:
            top_hit = ret_hits[0] if ret_hits else None
            c_intent = top_hit.get("silver_intent", predicted_intent) if (self.retrieval_policy == "global" and top_hit) else predicted_intent
            candidate_procedure = self._get_candidate_procedure(top_hit, c_intent, customer_text)
            resolution_check_result = self.resolution_checker.check_resolution(customer_text, candidate_procedure)

        # 4. Two-Layer Escalation Decision
        esc_result = decide_escalation(
            customer_text=customer_text,
            predicted_intent=predicted_intent,
            retrieval_similarity=top_sim,
            classifier_confidence=confidence,
            tau=self.tau,
            tau_low=self.tau_low,
            tau_high=self.tau_high,
            resolution_check_result=resolution_check_result
        )
        decision = esc_result["decision"]
        decision_reason = esc_result["decision_reason"]
        gate_triggered = esc_result["gate_triggered"]

        # 5. Grounded Reply Drafting (reuse existing draft if decision did not change)
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

def evaluate_agent(
    subsample: Optional[int] = 38,
    tau_low: Optional[float] = 0.65,
    tau_high: Optional[float] = 0.73,
    policy: str = "scoped",
    use_resolution_check: bool = True,
    tau: float = DEFAULT_TAU,
    use_cache: bool = True
):
    """Run agent evaluation over golden set or stratified subsample with incremental persistence."""
    golden_df = pd.read_csv(GOLDEN_PATH)
    
    if subsample is not None and subsample < len(golden_df):
        from sklearn.model_selection import train_test_split
        strat = golden_df["gold_intent"] + "_" + golden_df["difficulty_tier"]
        _, eval_df = train_test_split(
            golden_df,
            test_size=subsample,
            stratify=strat,
            random_state=42
        )
        eval_df = eval_df.reset_index(drop=True)
        out_csv_path = "/Users/kavya/Desktop/Groundcheck/src/agent_predictions_subsample.csv"
        scope_str = f"Stratified Subsample N={len(eval_df)}/151"
    else:
        eval_df = golden_df
        out_csv_path = AGENT_PREDICTIONS_CACHE
        scope_str = f"Full Golden Set N={len(eval_df)}"

    mode_str = f"tau_low={tau_low}, tau_high={tau_high}, policy={policy}, check={use_resolution_check}"
    print(f"\n========================================================")
    print(f"SPOTIFY SUPPORT AGENT EVALUATION ({scope_str}, {mode_str})")
    print(f"========================================================")

    y_true = eval_df["gold_decision"]
    easy_mask = eval_df["difficulty_tier"] == "easy"
    hard_mask = eval_df["difficulty_tier"] == "hard"

    cached_clf = {}
    if use_cache and os.path.exists(CLASSIFIER_PREDICTIONS_PATH):
        try:
            clf_df = pd.read_csv(CLASSIFIER_PREDICTIONS_PATH)
            for _, r in clf_df.iterrows():
                cached_clf[str(r["id"])] = (r["predicted_intent"], float(r.get("confidence", 1.0)))
            print(f"Loaded {len(cached_clf)} cached classifier predictions from disk.")
        except Exception:
            cached_clf = {}

    existing_records = {}
    if use_cache and os.path.exists(out_csv_path):
        try:
            cached_df = pd.read_csv(out_csv_path)
            for _, r in cached_df.iterrows():
                existing_records[str(r["id"])] = r.to_dict()
            print(f"Loaded {len(existing_records)} prior agent predictions from disk for smart draft reuse.")
        except Exception:
            existing_records = {}

    agent = SpotifySupportAgent(
        tau=tau,
        tau_low=tau_low,
        tau_high=tau_high,
        retrieval_policy=policy,
        use_resolution_check=use_resolution_check
    )
    print(f"Executing agent inference over {len(eval_df)} golden set threads ({mode_str})...")
    start_time = time.time()
    records = []

    for idx, (_, row) in enumerate(eval_df.iterrows()):
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
        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_df):
            temp_df = pd.DataFrame(records)
            temp_df.to_csv(out_csv_path, index=False)
            print(f"  Processed {idx + 1}/{len(eval_df)} rows ({time.time() - start_time:.1f}s)...")

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(out_csv_path, index=False)
    print(f"Saved agent predictions to {out_csv_path} ({time.time() - start_time:.1f}s).")

    y_pred = pred_df["decision"]
    acc_overall = accuracy_score(y_true, y_pred)
    acc_easy = accuracy_score(y_true[easy_mask], y_pred[easy_mask]) if easy_mask.sum() > 0 else 0.0
    acc_hard = accuracy_score(y_true[hard_mask], y_pred[hard_mask]) if hard_mask.sum() > 0 else 0.0

    fah = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "escalate" and yp == "auto_handle")
    fe = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "auto_handle" and yp == "escalate")
    fah_total = sum(y_true == "escalate")
    fe_total = sum(y_true == "auto_handle")

    if subsample is not None and subsample < len(golden_df):
        cost_sub = 4.0 * fah + 1.0 * fe
        print("\n--------------------------------------------------------")
        print(f"SUBSAMPLE EVALUATION REPORT (N={len(eval_df)} Stratified Rows)")
        print("--------------------------------------------------------")
        print(f"Overall Escalation Accuracy: {acc_overall*100:.2f}% ({sum(y_true == y_pred)}/{len(eval_df)})")
        print(f"  Easy Tier Accuracy (N={easy_mask.sum()}):  {acc_easy*100:.2f}% ({sum((y_true == y_pred) & easy_mask)}/{easy_mask.sum()})")
        print(f"  Hard Tier Accuracy (N={hard_mask.sum()}):  {acc_hard*100:.2f}% ({sum((y_true == y_pred) & hard_mask)}/{hard_mask.sum()})")
        print(f"False Auto-Handles (FAH):    {fah:2d}/{fah_total} ({fah/fah_total*100:4.2f}%)")
        print(f"False Escalations (FE):      {fe:2d}/{fe_total} ({fe/fe_total*100:4.2f}%)")
        print(f"Asymmetric Cost (4*FAH+1*FE): {cost_sub:.1f}")
    else:
        # Stratified Held-Out Evaluation Split (N=76)
        from sklearn.model_selection import train_test_split
        strat = pred_df["gold_intent"] + "_" + pred_df["difficulty_tier"]
        calib_df, held_out_df = train_test_split(
            pred_df,
            test_size=0.5,
            stratify=strat,
            random_state=42
        )
        y_true_ho = held_out_df["gold_decision"]
        y_pred_ho = held_out_df["decision"]
        acc_ho = accuracy_score(y_true_ho, y_pred_ho)
        fah_ho = sum(1 for yt, yp in zip(y_true_ho, y_pred_ho) if yt == "escalate" and yp == "auto_handle")
        fah_ho_total = sum(y_true_ho == "escalate")
        fe_ho = sum(1 for yt, yp in zip(y_true_ho, y_pred_ho) if yt == "auto_handle" and yp == "escalate")
        fe_ho_total = sum(y_true_ho == "auto_handle")
        cost_ho = 4.0 * fah_ho + 1.0 * fe_ho

        print("\n--------------------------------------------------------")
        print("PRIMARY HONEST ESTIMATE: HELD-OUT SPLIT (N=76)")
        print("--------------------------------------------------------")
        print(f"Overall Escalation Accuracy: {acc_ho*100:.2f}% ({sum(y_true_ho == y_pred_ho)}/{len(held_out_df)})")
        print(f"False Auto-Handles (FAH):    {fah_ho:2d}/{fah_ho_total} ({fah_ho/fah_ho_total*100:4.2f}%)")
        print(f"False Escalations (FE):      {fe_ho:2d}/{fe_ho_total} ({fe_ho/fe_ho_total*100:4.2f}%)")
        print(f"Asymmetric Cost (4*FAH+1*FE): {cost_ho:.1f}")

        print("\n--------------------------------------------------------")
        print("SECONDARY BOUND: FULL GOLDEN SET (N=151)")
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
    parser = argparse.ArgumentParser(description="Run SpotifySupportAgent evaluation over golden set or subsample.")
    parser.add_argument("--subsample", type=int, default=None, help="Evaluate on a stratified subsample of N rows (default: 38 unless --full)")
    parser.add_argument("--full", action="store_true", help="Evaluate over the full 151 golden set rows (~24 min runtime)")
    parser.add_argument("--tau-low", type=float, default=0.65, help="Lower similarity threshold for LLM verification (default: 0.65)")
    parser.add_argument("--tau-high", type=float, default=0.73, help="Upper similarity threshold (default: 0.73)")
    parser.add_argument("--policy", type=str, default="scoped", choices=["scoped", "global"], help="Retrieval partition policy (default: scoped)")
    parser.add_argument("--no-check", action="store_true", help="Disable LLM problem-resolution check (reverts to scalar cutoff)")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Fallback scalar threshold if thresholds unspecified")
    parser.add_argument("--no-cache", action="store_true", help="Force re-inference without cache")
    args = parser.parse_args()

    subsample_n = None if args.full else (args.subsample if args.subsample is not None else 38)

    evaluate_agent(
        subsample=subsample_n,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
        policy=args.policy,
        use_resolution_check=not args.no_check,
        tau=args.tau,
        use_cache=not args.no_cache
    )


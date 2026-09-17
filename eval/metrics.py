"""
eval/metrics.py

Phase 5 Component — Automated Evaluation Metrics Harness.
Calculates:
1. Intent Classification Metrics:
   - In-taxonomy (N=132) Overall Accuracy, Macro F1, Weighted F1
   - Per-intent Precision, Recall, F1, and Support
   - Difficulty tier breakdown (Easy vs Hard accuracy)
2. Escalation Decision Metrics:
   - Primary Honest Estimate: Stratified Held-Out Evaluation Split (N=76, tau=0.73)
   - Secondary Bound: Full Golden Set (N=151, tau=0.73, reflects calibration leakage)
   - False Auto-Handle Rate (FAH, safety risk) & False Escalation Rate (FE, queue burden)
   - Asymmetrically Weighted Escalation Cost: Cost = w_fah * FAH + w_fe * FE (default w_fah=4.0, w_fe=1.0)
   - Comparison against Baseline 1 (Always Escalate) and Baseline 3 (Locked TF-IDF tau=0.35)
3. Grounding & Retrieval Diagnostics:
   - Grounding source distribution (curated_sop vs retrieved_case vs none)
   - Layer 1 Safety Gate trigger frequency and precision
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

INTENT_CLASSES = [
    "playback_issue",
    "account_login",
    "premium_billing",
    "app_technical",
    "device_platform"
]


def compute_classification_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute intent classification metrics over in-taxonomy golden set rows.
    Out-of-taxonomy rows ('other') are excluded from 5-way classifier scoring.
    """
    # Filter to in-taxonomy rows
    in_tax = df[df["gold_intent"].isin(INTENT_CLASSES)].copy()
    y_true = in_tax["gold_intent"].values
    y_pred = in_tax["predicted_intent"].values

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-intent metrics
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENT_CLASSES, zero_division=0
    )
    per_intent = {}
    for i, intent in enumerate(INTENT_CLASSES):
        per_intent[intent] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
            "correct": int(np.sum((y_true == intent) & (y_pred == intent)))
        }

    # Difficulty tier breakdown
    tier_metrics = {}
    for tier in ["easy", "hard"]:
        sub = in_tax[in_tax["difficulty_tier"] == tier]
        if len(sub) > 0:
            t_acc = float(accuracy_score(sub["gold_intent"], sub["predicted_intent"]))
            tier_metrics[tier] = {
                "accuracy": t_acc,
                "correct": int((sub["gold_intent"] == sub["predicted_intent"]).sum()),
                "total": len(sub)
            }

    return {
        "sample_size": len(in_tax),
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_intent": per_intent,
        "by_difficulty_tier": tier_metrics
    }


def compute_escalation_split_metrics(
    sub_df: pd.DataFrame,
    w_fah: float = 4.0,
    w_fe: float = 1.0
) -> Dict[str, Any]:
    """Compute escalation accuracy, FAH, FE, and asymmetric cost on a specified subset."""
    total = len(sub_df)
    correct = int((sub_df["decision"] == sub_df["gold_decision"]).sum())
    acc = float(correct / total) if total > 0 else 0.0

    # False Auto-Handles (Gold = escalate, Predicted = auto_handle) -> Unsafe advice risk
    gold_esc = sub_df[sub_df["gold_decision"] == "escalate"]
    fah_count = int((gold_esc["decision"] == "auto_handle").sum())
    fah_total = len(gold_esc)
    fah_rate = float(fah_count / fah_total) if fah_total > 0 else 0.0

    # False Escalations (Gold = auto_handle, Predicted = escalate) -> Human routing overhead
    gold_auto = sub_df[sub_df["gold_decision"] == "auto_handle"]
    fe_count = int((gold_auto["decision"] == "escalate").sum())
    fe_total = len(gold_auto)
    fe_rate = float(fe_count / fe_total) if fe_total > 0 else 0.0

    # Asymmetric weighted cost (count-based: 4*FAH + 1*FE)
    asymmetric_cost = float(w_fah * fah_count + w_fe * fe_count)

    # Breakdown by difficulty tier
    tier_breakdown = {}
    for tier in ["easy", "hard"]:
        t_df = sub_df[sub_df["difficulty_tier"] == tier]
        if len(t_df) > 0:
            t_correct = int((t_df["decision"] == t_df["gold_decision"]).sum())
            t_esc = t_df[t_df["gold_decision"] == "escalate"]
            t_fah = int((t_esc["decision"] == "auto_handle").sum())
            t_auto = t_df[t_df["gold_decision"] == "auto_handle"]
            t_fe = int((t_auto["decision"] == "escalate").sum())

            tier_breakdown[tier] = {
                "total": len(t_df),
                "accuracy": float(t_correct / len(t_df)),
                "correct": t_correct,
                "fah_count": t_fah,
                "fah_total": len(t_esc),
                "fah_rate": float(t_fah / len(t_esc)) if len(t_esc) > 0 else 0.0,
                "fe_count": t_fe,
                "fe_total": len(t_auto),
                "fe_rate": float(t_fe / len(t_auto)) if len(t_auto) > 0 else 0.0
            }

    return {
        "sample_size": total,
        "overall_accuracy": acc,
        "correct": correct,
        "fah_count": fah_count,
        "fah_total": fah_total,
        "fah_rate": fah_rate,
        "fe_count": fe_count,
        "fe_total": fe_total,
        "fe_rate": fe_rate,
        "asymmetric_cost": asymmetric_cost,
        "w_fah": w_fah,
        "w_fe": w_fe,
        "by_difficulty_tier": tier_breakdown
    }


def compute_escalation_metrics(
    df: pd.DataFrame,
    test_size: float = 0.5,
    random_state: int = 42,
    w_fah: float = 4.0,
    w_fe: float = 1.0
) -> Dict[str, Any]:
    """
    Compute escalation metrics across both the held-out calibration split
    (primary honest estimate) and the full 151-row golden set (secondary bound).
    """
    # 1. Stratified Held-out Evaluation Split (Primary Honest Estimate)
    strat = df["gold_intent"] + "_" + df["difficulty_tier"]
    calib_df, held_out_df = train_test_split(
        df,
        test_size=test_size,
        stratify=strat,
        random_state=random_state
    )

    primary_held_out = compute_escalation_split_metrics(held_out_df, w_fah=w_fah, w_fe=w_fe)
    calibration_subset = compute_escalation_split_metrics(calib_df, w_fah=w_fah, w_fe=w_fe)
    secondary_full = compute_escalation_split_metrics(df, w_fah=w_fah, w_fe=w_fe)

    # Baselines for direct benchmarking (N=151)
    baseline1_always_esc = {
        "overall_accuracy": 68 / 151,
        "fah_rate": 0.0,
        "fah_count": 0,
        "fah_total": 68,
        "fe_rate": 1.0,
        "fe_count": 83,
        "fe_total": 83,
        "asymmetric_cost": float(w_fah * 0 + w_fe * 83)
    }
    baseline3_locked_hurdle = {
        "overall_accuracy": 80 / 151,
        "fah_rate": 27 / 68,
        "fah_count": 27,
        "fah_total": 68,
        "fe_rate": 44 / 83,
        "fe_count": 44,
        "fe_total": 83,
        "asymmetric_cost": float(w_fah * 27 + w_fe * 44)
    }

    return {
        "primary_honest_estimate_held_out": primary_held_out,
        "calibration_subset": calibration_subset,
        "secondary_bound_full_set": secondary_full,
        "baselines": {
            "baseline_1_always_escalate": baseline1_always_esc,
            "baseline_3_locked_hurdle": baseline3_locked_hurdle
        }
    }


def compute_retrieval_and_grounding_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Diagnostics on grounding sources, gate activations, and confidence distributions."""
    grounding_counts = df["grounding_source"].value_counts().to_dict()
    gate_counts = df["gate_triggered"].dropna().value_counts().to_dict()

    auto_handles = df[df["decision"] == "auto_handle"]
    escalates = df[df["decision"] == "escalate"]

    auto_conf = float(auto_handles["calibrated_confidence"].mean()) if len(auto_handles) > 0 else 0.0
    esc_conf = float(escalates["calibrated_confidence"].mean()) if len(escalates) > 0 else 0.0

    return {
        "grounding_source_distribution": grounding_counts,
        "gate_triggered_distribution": gate_counts,
        "mean_confidence_auto_handle": auto_conf,
        "mean_confidence_escalate": esc_conf,
        "total_rows": len(df)
    }


def print_evaluation_report(metrics: Dict[str, Any]) -> None:
    """Print beautifully formatted evaluation report to stdout."""
    print("=" * 78)
    print("SPOTIFYCARES AI SUPPORT AGENT — AUTOMATED EVALUATION METRICS REPORT")
    print("=" * 78)

    # 1. Classification
    clf = metrics["classification"]
    print("\n--- 1. INTENT CLASSIFICATION PERFORMANCE (In-Taxonomy N=132) ---")
    print(f"Overall Accuracy : {clf['overall_accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score   : {clf['macro_f1']:.3f}")
    print(f"Weighted F1 Score: {clf['weighted_f1']:.3f}")
    print(f"Easy Tier Acc    : {clf['by_difficulty_tier']['easy']['accuracy'] * 100:.2f}% ({clf['by_difficulty_tier']['easy']['correct']}/{clf['by_difficulty_tier']['easy']['total']})")
    print(f"Hard Tier Acc    : {clf['by_difficulty_tier']['hard']['accuracy'] * 100:.2f}% ({clf['by_difficulty_tier']['hard']['correct']}/{clf['by_difficulty_tier']['hard']['total']})")

    print("\nPer-Intent Metrics:")
    print(f"{'Intent':<20} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Support':<7} | {'Correct':<7}")
    print("-" * 65)
    for intent, vals in clf["per_intent"].items():
        print(f"{intent:<20} | {vals['precision']:<7.3f} | {vals['recall']:<7.3f} | {vals['f1']:<7.3f} | {vals['support']:<7} | {vals['correct']:<7}")

    # 2. Escalation
    esc = metrics["escalation"]
    p_held = esc["primary_honest_estimate_held_out"]
    s_full = esc["secondary_bound_full_set"]
    b1 = esc["baselines"]["baseline_1_always_escalate"]
    b3 = esc["baselines"]["baseline_3_locked_hurdle"]

    print("\n--- 2. ESCALATION DECISION PERFORMANCE & ASYMMETRIC LOSS ---")
    print(f"{'Metric':<32} | {'Baseline 1':<12} | {'Baseline 3':<12} | {'Delivered Held-Out (N=76)':<26} | {'Delivered Full Set (N=151)*':<28}")
    print("-" * 122)
    b1_fah = f"{b1['fah_count']}/{b1['fah_total']}"
    b3_fah = f"{b3['fah_count']}/{b3['fah_total']}"
    p_held_fah = f"{p_held['fah_count']}/{p_held['fah_total']}"
    s_full_fah = f"{s_full['fah_count']}/{s_full['fah_total']}"

    b1_fe = f"{b1['fe_count']}/{b1['fe_total']}"
    b3_fe = f"{b3['fe_count']}/{b3['fe_total']}"
    p_held_fe = f"{p_held['fe_count']}/{p_held['fe_total']}"
    s_full_fe = f"{s_full['fe_count']}/{s_full['fe_total']}"

    print(f"{'Overall Accuracy':<32} | {b1['overall_accuracy']*100:>10.2f}% | {b3['overall_accuracy']*100:>10.2f}% | {p_held['overall_accuracy']*100:>24.2f}% | {s_full['overall_accuracy']*100:>26.2f}%")
    print(f"{'False Auto-Handles (FAH)':<32} | {b1['fah_rate']*100:>10.2f}% | {b3['fah_rate']*100:>10.2f}% | {p_held['fah_rate']*100:>24.2f}% | {s_full['fah_rate']*100:>26.2f}%")
    print(f"{'  FAH Count / Total':<32} | {b1_fah:>12} | {b3_fah:>12} | {p_held_fah:>26} | {s_full_fah:>28}")
    print(f"{'False Escalations (FE)':<32} | {b1['fe_rate']*100:>10.2f}% | {b3['fe_rate']*100:>10.2f}% | {p_held['fe_rate']*100:>24.2f}% | {s_full['fe_rate']*100:>26.2f}%")
    print(f"{'  FE Count / Total':<32} | {b1_fe:>12} | {b3_fe:>12} | {p_held_fe:>26} | {s_full_fe:>28}")
    print(f"{'Asymmetric Cost (4*FAH + 1*FE)':<32} | {b1['asymmetric_cost']:>12.1f} | {b3['asymmetric_cost']:>12.1f} | {p_held['asymmetric_cost']:>26.1f} | {s_full['asymmetric_cost']:>28.1f}")
    print("*Note: Delivered Full Set includes the calibration split (optimistic bound); Held-Out (N=76) is primary.")

    print("\nHeld-Out Split Difficulty Breakdown:")
    easy_h = p_held["by_difficulty_tier"]["easy"]
    hard_h = p_held["by_difficulty_tier"]["hard"]
    print(f"  Easy Tier Acc: {easy_h['accuracy']*100:.2f}% ({easy_h['correct']}/{easy_h['total']}), FAH: {easy_h['fah_count']}/{easy_h['fah_total']}, FE: {easy_h['fe_count']}/{easy_h['fe_total']}")
    print(f"  Hard Tier Acc: {hard_h['accuracy']*100:.2f}% ({hard_h['correct']}/{hard_h['total']}), FAH: {hard_h['fah_count']}/{hard_h['fah_total']}, FE: {hard_h['fe_count']}/{hard_h['fe_total']}")

    # 3. Grounding & Diagnostics
    diag = metrics["diagnostics"]
    print("\n--- 3. GROUNDING & ROUTING DIAGNOSTICS (N=151) ---")
    print(f"Grounding Sources: {diag['grounding_source_distribution']}")
    print(f"Gate Triggers    : {diag['gate_triggered_distribution']}")
    print(f"Mean Conf (Auto) : {diag['mean_confidence_auto_handle']:.3f}")
    print(f"Mean Conf (Esc)  : {diag['mean_confidence_escalate']:.3f}")
    print("=" * 78)


def run_full_evaluation(
    predictions_path: str = "src/agent_predictions_golden151.csv",
    output_json: Optional[str] = None
) -> Dict[str, Any]:
    """Execute full evaluation pipeline and optionally export JSON."""
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found at: {predictions_path}")

    df = pd.read_csv(predictions_path)

    classification_metrics = compute_classification_metrics(df)
    escalation_metrics = compute_escalation_metrics(df)
    diagnostics = compute_retrieval_and_grounding_metrics(df)

    results = {
        "classification": classification_metrics,
        "escalation": escalation_metrics,
        "diagnostics": diagnostics
    }

    print_evaluation_report(results)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved structured metrics to: {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SpotifyCares AI Support Agent")
    parser.add_argument(
        "--predictions",
        default="src/agent_predictions_golden151.csv",
        help="Path to agent predictions CSV"
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional path to export JSON metrics"
    )
    args = parser.parse_args()
    run_full_evaluation(predictions_path=args.predictions, output_json=args.json)

"""
eval/validate_judge.py

Phase 5 Component — LLM Judge Validation Against Human Golden Labels.
Evaluates:
1. Human-Judge Decision Agreement (% agreement & Cohen's Kappa with 95% Confidence Interval)
2. Disaggregated Agreement by Difficulty Tier (Easy vs Hard) with small-N caveat
3. Score Correlation & MAE across Groundedness (1-5) and Tone (1-5)
4. Qualitative Error Analysis isolating specific Judge Failure Patterns

Model Independence:
- Defaults to 'openai/gpt-oss-120b' to eliminate same-model judge bias against drafter 'qwen/qwen3.8-27b'.
- Supports '--model' flag to compare architectures.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from eval.judge import SupportReplyJudge, PRIMARY_JUDGE_MODEL, FALLBACK_JUDGE_MODEL


def compute_kappa_with_ci(y_true, y_pred, labels=["pass", "fail"]) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute Cohen's Kappa along with asymptotic standard error and 95% Confidence Interval.
    Formula: SE = sqrt( Po * (1 - Po) / (N * (1 - Pe)^2) )
    """
    k = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = float(np.sum(cm))
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0)

    po = float(np.trace(cm) / n)
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    pe = float(np.sum(row_sums * col_sums) / (n * n))

    if pe >= 1.0 or po == 1.0:
        se = 0.0
    else:
        denom = n * ((1.0 - pe) ** 2)
        se = float(np.sqrt(max(0.0, (po * (1.0 - po)) / denom))) if denom > 0 else 0.0

    ci_low = max(-1.0, k - 1.96 * se)
    ci_high = min(1.0, k + 1.96 * se)
    return k, se, (ci_low, ci_high)


def run_judge_validation(
    audit_path: str = "eval/human_audit_50.csv",
    output_path: str = "eval/judge_validation_results.csv",
    model_name: str = PRIMARY_JUDGE_MODEL,
    force: bool = False
) -> pd.DataFrame:
    """Run judge evaluation over human-audited validation rows (or load cache)."""
    if not os.path.exists(audit_path):
        raise FileNotFoundError(f"Human audit file not found at: {audit_path}")

    audit_df = pd.read_csv(audit_path)

    # Check if cached results exist and match the requested model
    if os.path.exists(output_path) and not force:
        cached_df = pd.read_csv(output_path)
        if len(cached_df) == len(audit_df) and "judge_model" in cached_df.columns:
            if cached_df["judge_model"].iloc[0] == model_name:
                print(f"Loading cached judge validation results from {output_path} ({model_name})...")
                return cached_df

    print(f"Executing LLM Judge evaluation across {len(audit_df)} validation rows using model: {model_name}...")
    judge = SupportReplyJudge(model_name=model_name)
    judge_results = []

    for idx, row in audit_df.iterrows():
        res = judge.evaluate_reply(
            row_id=row["id"],
            customer_text=row["customer_text"],
            predicted_intent=row.get("predicted_intent", "playback_issue"),
            decision=row["decision"],
            drafted_reply=row["drafted_reply"],
            evidence_used=str(row["evidence_used"]),
            difficulty_tier=row["difficulty_tier"]
        )
        judge_results.append(res)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(audit_df):
            print(f"  Processed {idx + 1}/{len(audit_df)} rows...")

    judge_df = pd.DataFrame(judge_results)
    merged_df = audit_df.merge(judge_df, on=["id", "difficulty_tier"])
    merged_df.to_csv(output_path, index=False)
    print(f"Saved validation evaluation results to: {output_path}")
    return merged_df


def analyze_agreement(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall and tier-disaggregated agreement statistics."""
    # 1. Overall Decision Agreement
    overall_agree = float((df["human_decision"] == df["judge_decision"]).mean())
    k_all, se_all, ci_all = compute_kappa_with_ci(df["human_decision"], df["judge_decision"])

    # 2. Disaggregated by Difficulty Tier
    tier_stats = {}
    for tier in ["easy", "hard"]:
        sub = df[df["difficulty_tier"] == tier]
        t_agree = float((sub["human_decision"] == sub["judge_decision"]).mean())
        k_t, se_t, ci_t = compute_kappa_with_ci(sub["human_decision"], sub["judge_decision"])
        tier_stats[tier] = {
            "n": len(sub),
            "agreement_rate": t_agree,
            "agreed_count": int((sub["human_decision"] == sub["judge_decision"]).sum()),
            "cohen_kappa": k_t,
            "standard_error": se_t,
            "ci_95": ci_t
        }

    # 3. Continuous Score Metrics (Groundedness & Tone)
    ground_mae = float(np.mean(np.abs(df["human_groundedness"] - df["groundedness_score"])))
    tone_mae = float(np.mean(np.abs(df["human_tone"] - df["tone_score"])))

    ground_spearman, _ = spearmanr(df["human_groundedness"], df["groundedness_score"])
    tone_spearman, _ = spearmanr(df["human_tone"], df["tone_score"])

    # 4. Discordance Analysis (Human != Judge)
    discordant = df[df["human_decision"] != df["judge_decision"]].copy()

    return {
        "sample_size": len(df),
        "overall_agreement_rate": overall_agree,
        "overall_agreed_count": int((df["human_decision"] == df["judge_decision"]).sum()),
        "overall_cohen_kappa": k_all,
        "overall_standard_error": se_all,
        "overall_ci_95": ci_all,
        "by_difficulty_tier": tier_stats,
        "groundedness_mae": ground_mae,
        "groundedness_spearman": float(ground_spearman) if not np.isnan(ground_spearman) else 0.0,
        "tone_mae": tone_mae,
        "tone_spearman": float(tone_spearman) if not np.isnan(tone_spearman) else 0.0,
        "discordant_cases_count": len(discordant),
        "discordant_df": discordant
    }


def print_validation_report(results: Dict[str, Any], model_name: str) -> None:
    """Print comprehensive validation report highlighting tier differences and failure modes."""
    print("\n" + "=" * 82)
    print("PHASE 5: LLM-AS-A-JUDGE VALIDATION REPORT (HUMAN AUDIT N=50)")
    print(f"Evaluator Model: {model_name} (Cross-Architecture Independent Judge)")
    print("=" * 82)

    print("\n--- 1. HEADLINE AGREEMENT METRICS ---")
    print(f"Overall Raw Agreement: {results['overall_agreement_rate'] * 100:.2f}% ({results['overall_agreed_count']}/{results['sample_size']})")
    ci_str = f"[{results['overall_ci_95'][0]:.3f}, {results['overall_ci_95'][1]:.3f}]"
    print(f"Overall Cohen's Kappa: {results['overall_cohen_kappa']:.3f} (SE = {results['overall_standard_error']:.3f}, 95% CI: {ci_str})")
    print(f"Groundedness MAE     : {results['groundedness_mae']:.3f} (Spearman rho = {results['groundedness_spearman']:.3f})")
    print(f"Tone Score MAE       : {results['tone_mae']:.3f} (Spearman rho = {results['tone_spearman']:.3f})")

    print("\n--- 2. DISAGGREGATED AGREEMENT BY DIFFICULTY TIER (SMALL-N ANALYSIS) ---")
    easy = results["by_difficulty_tier"]["easy"]
    hard = results["by_difficulty_tier"]["hard"]

    print(f"{'Tier':<10} | {'N':<5} | {'Raw Agreement':<16} | {'Cohen Kappa (k)':<18} | {'Std Error':<11} | {'95% Confidence Interval':<24}")
    print("-" * 92)
    easy_ci = f"[{easy['ci_95'][0]:.3f}, {easy['ci_95'][1]:.3f}]"
    hard_ci = f"[{hard['ci_95'][0]:.3f}, {hard['ci_95'][1]:.3f}]"
    easy_agr_str = f"{easy['agreement_rate']*100:.2f}% ({easy['agreed_count']}/{easy['n']})"
    hard_agr_str = f"{hard['agreement_rate']*100:.2f}% ({hard['agreed_count']}/{hard['n']})"
    print(f"{'Easy Tier':<10} | {easy['n']:<5} | {easy_agr_str:<16} | {easy['cohen_kappa']:<18.3f} | {easy['standard_error']:<11.3f} | {easy_ci:<24}")
    print(f"{'Hard Tier':<10} | {hard['n']:<5} | {hard_agr_str:<16} | {hard['cohen_kappa']:<18.3f} | {hard['standard_error']:<11.3f} | {hard_ci:<24}")

    print("\n> [!CAUTION]")
    print("> Small-N Caveat: With N_easy = 20 and N_hard = 30, asymptotic standard errors are substantial")
    print(f"> (SE_easy = {easy['standard_error']:.3f}, SE_hard = {hard['standard_error']:.3f}). The confidence intervals span a wide band.")
    print("> Point estimates should NOT be interpreted as high-precision bounds; they indicate relative directional trends.")

    # 3. Error Analysis
    disc = results["discordant_df"]
    print(f"\n--- 3. DISCORDANCE & JUDGE FAILURE PATTERN ANALYSIS ({len(disc)} Cases) ---")
    if len(disc) == 0:
        print("Zero disagreements between human and judge.")
    else:
        print(f"Disagreements: {len(disc)} cases where Human Decision != Judge Decision.\n")
        for i, (_, row) in enumerate(disc.iterrows(), 1):
            print(f"Case {i}: [{row['id']}] Tier: {row['difficulty_tier']} | Human: {row['human_decision'].upper()} (G={row['human_groundedness']}) vs Judge: {row['judge_decision'].upper()} (G={row['groundedness_score']})")
            print(f"  Customer : \"{row['customer_text'][:90]}\"")
            print(f"  Draft    : \"{row['drafted_reply'][:90]}\"")
            print(f"  Human Note: {row['human_notes']}")
            print(f"  Judge Note: {row['judge_notes']}")
            print("-" * 78)

    print("=" * 82)


def main():
    parser = argparse.ArgumentParser(description="Validate LLM Judge against Human Golden Set")
    parser.add_argument("--audit_path", default="eval/human_audit_50.csv", help="Path to human audit CSV")
    parser.add_argument("--output_path", default="eval/judge_validation_results.csv", help="Output path for validation results")
    parser.add_argument("--model", default=PRIMARY_JUDGE_MODEL, help="Judge model name")
    parser.add_argument("--force", action="store_true", help="Force recomputation ignoring cache")
    args = parser.parse_args()

    df = run_judge_validation(
        audit_path=args.audit_path,
        output_path=args.output_path,
        model_name=args.model,
        force=args.force
    )
    analysis = analyze_agreement(df)
    print_validation_report(analysis, args.model)


if __name__ == "__main__":
    main()

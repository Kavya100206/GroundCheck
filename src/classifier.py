"""
src/classifier.py

Phase 3 — LLM Intent Classifier for SpotifyCares Customer Support.
Uses Groq API with few-shot prompting using training-pool examples only (zero golden set leakage).

Evaluated against the 132 in-taxonomy golden rows (Option A / Decision Log Entry 13).
Target hurdle to beat: Baseline 2 (TF-IDF + LogReg) at 66.67% accuracy and 0.630 macro F1.
"""

import os
import re
import json
import time
import socket
import argparse
import pandas as pd
from typing import Dict, Any, List, Optional
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Robust DNS resolution patch for macOS network environment
DNS_CACHE = {'api.groq.com': '104.18.38.236'}
_orig_getaddrinfo = socket.getaddrinfo

def custom_getaddrinfo(host, port, *args, **kwargs):
    if host in DNS_CACHE:
        return _orig_getaddrinfo(DNS_CACHE[host], port, *args, **kwargs)
    return _orig_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = custom_getaddrinfo

GOLDEN_PATH = "/Users/kavya/Desktop/Groundcheck/golden_set_to_label.csv"
ENV_PATH = "/Users/kavya/Desktop/Groundcheck/.env"
PREDICTIONS_CACHE_PATH = "/Users/kavya/Desktop/Groundcheck/src/classifier_predictions.csv"

def load_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key and os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            for line in f:
                if line.strip().startswith("GROQ_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip()
                    break
    if not key:
        raise ValueError("GROQ_API_KEY not found in environment or .env file.")
    return key

# System Prompt & Few-Shot Examples (strictly drawn from training pool, zero golden set)
SYSTEM_PROMPT = """You are an expert customer support intent classifier for SpotifyCares.
Your task is to classify incoming customer tweets into EXACTLY ONE of the following 5 intents:

1. playback_issue: Audio playback fails, skips, buffers, pauses unexpectedly, sound is distorted, or music stops playing.
2. app_technical: The Spotify app crashes on launch, freezes, shows a black/blank screen, won't install/update, or requires clean reinstall/clearing cache.
3. account_login: Login authentication failures, password resets, locked accounts, 2FA/verification codes, or username/email access recovery.
4. device_platform: Platform- or device-specific integration and compatibility bugs (e.g. PlayStation PS4/PS5, Xbox, Smart TVs, Apple/Android CarPlay, Amazon Echo/Alexa, browser web player, or OS lock screen media controls).
5. premium_billing: Subscription charges, unexpected duplicate fees, billing disputes, refund requests, Family/Duo plan management, or cancellation payments.

Output Format:
You must respond ONLY with a raw JSON object (no markdown code blocks, no preamble, no thinking) containing:
{
  "intent": "<playback_issue|app_technical|account_login|device_platform|premium_billing>",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one concise sentence explaining the routing decision>"
}"""

FEW_SHOT_EXAMPLES = [
    {
        "customer_text": "Why does my music keep randomly pausing every 20 seconds? It's happening on every single track today.",
        "intent": "playback_issue",
        "confidence": 0.95,
        "reason": "Audio playback pauses repeatedly and unexpectedly across songs."
    },
    {
        "customer_text": "Every time I tap the Spotify app icon on my phone, it crashes immediately back to home screen. Won't open.",
        "intent": "app_technical",
        "confidence": 0.98,
        "reason": "Application crashes upon launch and fails to open."
    },
    {
        "customer_text": "I was locked out of my account and I never received the password reset link to my email.",
        "intent": "account_login",
        "confidence": 0.95,
        "reason": "Customer is locked out and password recovery email is not being delivered."
    },
    {
        "customer_text": "My PS4 isn't appearing as an available Connect device on my phone anymore, even though both are on the same WiFi.",
        "intent": "device_platform",
        "confidence": 0.95,
        "reason": "Device discovery and Spotify Connect integration failure on PlayStation 4."
    },
    {
        "customer_text": "I was charged twice on my credit card this month for my subscription. I need a refund for the extra charge.",
        "intent": "premium_billing",
        "confidence": 0.98,
        "reason": "Billing dispute regarding unauthorized duplicate subscription charge requesting refund."
    },
    {
        "customer_text": "After the latest OS update, the lockscreen playback controls and widget completely disappeared on my phone.",
        "intent": "device_platform",
        "confidence": 0.92,
        "reason": "Operating system level media control widget integration failure."
    }
]

def build_few_shot_messages(customer_text: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"Customer tweet: \"{ex['customer_text']}\""
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "intent": ex["intent"],
                "confidence": ex["confidence"],
                "reason": ex["reason"]
            })
        })
    messages.append({
        "role": "user",
        "content": f"Customer tweet: \"{customer_text}\""
    })
    return messages

class IntentClassifier:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        self.api_key = load_groq_key()
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        self.model_name = model_name

    def classify(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
        messages = build_few_shot_messages(text)
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=80,
                    response_format={"type": "json_object"}
                )
                raw_content = resp.choices[0].message.content.strip()
                parsed = json.loads(raw_content)
                intent = parsed.get("intent", "").lower().strip()
                valid_intents = {"playback_issue", "app_technical", "account_login", "device_platform", "premium_billing"}
                if intent not in valid_intents:
                    # Fallback pattern match if model emitted slightly perturbed string
                    for v in valid_intents:
                        if v in intent:
                            parsed["intent"] = v
                            break
                return parsed
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    return {
                        "intent": "playback_issue", # fallback
                        "confidence": 0.20,
                        "reason": f"Classification error: {str(e)}"
                    }

def evaluate_classifier(model_name: str = "qwen/qwen3.8-27b", use_cache: bool = True):
    print(f"Loading golden set from {GOLDEN_PATH}...")
    golden_df = pd.read_csv(GOLDEN_PATH)
    in_tax_df = golden_df[golden_df["gold_intent"] != "other"].copy()
    print(f"Evaluating on {len(in_tax_df)} in-taxonomy golden rows (Option A locked policy)...")

    predictions = []
    cached_df = None
    if use_cache and os.path.exists(PREDICTIONS_CACHE_PATH):
        try:
            cached_df = pd.read_csv(PREDICTIONS_CACHE_PATH)
            if len(cached_df) == len(in_tax_df):
                print(f"Loaded {len(cached_df)} cached predictions from {PREDICTIONS_CACHE_PATH}")
        except Exception:
            cached_df = None

    if cached_df is None:
        classifier = IntentClassifier(model_name=model_name)
        print(f"Running LLM inference with model: {model_name}...")
        start_time = time.time()
        for idx, (_, row) in enumerate(in_tax_df.iterrows()):
            res = classifier.classify(row["customer_text"])
            predictions.append({
                "id": row["id"],
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "gold_intent": row["gold_intent"],
                "difficulty_tier": row["difficulty_tier"],
                "predicted_intent": res.get("intent"),
                "confidence": res.get("confidence", 0.0),
                "reason": res.get("reason", "")
            })
            if (idx + 1) % 20 == 0 or (idx + 1) == len(in_tax_df):
                print(f"  Processed {idx + 1}/{len(in_tax_df)} rows ({time.time() - start_time:.1f}s)...")
        
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(PREDICTIONS_CACHE_PATH, index=False)
        print(f"Saved predictions to {PREDICTIONS_CACHE_PATH}")
    else:
        pred_df = cached_df

    y_true = pred_df["gold_intent"]
    y_pred = pred_df["predicted_intent"]

    acc = accuracy_score(y_true, y_pred)
    easy_mask = pred_df["difficulty_tier"] == "easy"
    hard_mask = pred_df["difficulty_tier"] == "hard"
    acc_easy = accuracy_score(y_true[easy_mask], y_pred[easy_mask])
    acc_hard = accuracy_score(y_true[hard_mask], y_pred[hard_mask])

    print("\n========================================================")
    print("PHASE 3 LLM CLASSIFIER EVALUATION RESULTS")
    print(f"Model: {model_name} (Few-Shot Prompted, Zero Golden Leakage)")
    print("========================================================")
    print(f"Overall In-Taxonomy Accuracy (N=132): {acc*100:.2f}% (Baseline 2 hurdle: 66.67%)")
    print(f"  Easy Tier Accuracy (N={easy_mask.sum()}):  {acc_easy*100:.2f}%")
    print(f"  Hard Tier Accuracy (N={hard_mask.sum()}):  {acc_hard*100:.2f}%\n")

    report = classification_report(y_true, y_pred, digits=3)
    print(report)

    print("\nConfusion Matrix:")
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True:{l}" for l in labels], columns=[f"Pred:{l}" for l in labels])
    print(cm_df)

    return {
        "accuracy": acc,
        "acc_easy": acc_easy,
        "acc_hard": acc_hard,
        "report": report,
        "pred_df": pred_df
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b", help="Groq model name")
    parser.add_argument("--no-cache", action="store_true", help="Force re-inference without cache")
    args = parser.parse_args()

    evaluate_classifier(model_name=args.model, use_cache=not args.no_cache)

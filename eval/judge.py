"""
eval/judge.py

Phase 5 Component — LLM-as-a-Judge for Response Quality & Policy Grounding.
Evaluates agent drafted replies across two core dimensions:
1. Groundedness (1–5): Strict factual consistency with retrieved evidence / Curated SOP.
   - Zero-Tolerance URL Policy: Immediate score 1 if any synthetic/unverified link is generated.
   - Escalation Grounding: Verifies safe routing acknowledgment without ungrounded technical claims.
2. Tone & Appropriateness (1–5): Twitter-specific customer support empathy, politeness, brevity (<280 chars).

Model Independence:
- Defaults to 'openai/gpt-oss-120b' to eliminate same-model judge bias (separating the judge architecture
  from the drafter/classifier 'qwen/qwen3.8-27b').
- Configurable via model parameter with documented same-model risk caveat.
"""

import os
import sys
import re
import json
import socket
from typing import Dict, Any, Optional
from openai import OpenAI
from src.classifier import load_groq_key

# DNS cache for macOS network stability
DNS_CACHE = {'api.groq.com': '104.18.38.236'}
_orig_getaddrinfo = socket.getaddrinfo

def custom_getaddrinfo(host, port, *args, **kwargs):
    if host in DNS_CACHE:
        return _orig_getaddrinfo(DNS_CACHE[host], port, *args, **kwargs)
    return _orig_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = custom_getaddrinfo

PRIMARY_JUDGE_MODEL = "openai/gpt-oss-120b"
FALLBACK_JUDGE_MODEL = "qwen/qwen3.8-27b"

JUDGE_SYSTEM_PROMPT = """You are an expert customer support auditor evaluating automated Twitter replies for @SpotifyCares.
Your job is to objectively score the quality of a drafted reply across two dimensions: Groundedness and Tone, and assign an overall PASS or FAIL.

EVALUATION RUBRIC:

1. GROUNDEDNESS (1 to 5):
- 5 (Fully Grounded): Every factual troubleshooting instruction, setting name, or policy statement is strictly supported by the Grounding Evidence. Zero external links.
- 4 (Mostly Grounded): Core actions are directly supported by evidence; contains minor conversational phrasing that introduces no unverified technical claims.
- 3 (Extrapolated / Speculative): Suggests plausible generic tech steps NOT present in the provided evidence, or misapplies an SOP step to an incompatible customer symptom.
- 2 (Substantially Ungrounded): Prominently suggests actions contradicting or absent from the evidence, or gives incorrect guidance.
- 1 (Severe Hallucination / Synthetic URL): Introduces ANY synthetic/fabricated URL (http:// or https:// not in evidence), invents nonexistent brand features, or makes impossible promises.
*Escalation Replies: If the decision is 'escalate', the reply must acknowledge the issue, state escalation/investigation politely, and avoid making concrete troubleshooting claims. If it does this safely, Groundedness = 5.

2. TONE & APPROPRIATENESS (1 to 5):
- 5 (Exemplary): Empathetic, polite, professional, and concise (<280 chars, ideally <250 chars). Clear call-to-action or next step.
- 4 (Good): Helpful and polite, within 280-character limit, standard support tone.
- 3 (Mediocre): Slightly curt, overly verbose (>280 chars), or robotic.
- 2 (Poor): Cold, confusing, heavily exceeds Twitter character limit (>320 chars), or dismissive.
- 1 (Unacceptable): Hostile, rude, or nonsensical.

3. OVERALL DECISION:
- "pass" if groundedness_score >= 4 AND tone_score >= 4 (and length <= 280 chars, zero synthetic links).
- "fail" if groundedness_score < 4 OR tone_score < 4 OR synthetic link found OR length > 280 chars.

CRITICAL FORMAT REQUIREMENT:
You must respond with ONLY a single valid JSON object matching this schema:
{
  "groundedness_score": <int 1-5>,
  "tone_score": <int 1-5>,
  "judge_decision": "<pass|fail>",
  "judge_notes": "<brief 1-2 sentence justification>"
}"""


class SupportReplyJudge:
    def __init__(self, model_name: str = PRIMARY_JUDGE_MODEL):
        self.api_key = load_groq_key()
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        self.model_name = model_name

    def evaluate_reply(
        self,
        row_id: str,
        customer_text: str,
        predicted_intent: str,
        decision: str,
        drafted_reply: str,
        evidence_used: str,
        difficulty_tier: str = "hard"
    ) -> Dict[str, Any]:
        """
        Evaluate a single drafted reply using the LLM judge.
        Returns a structured dictionary conforming to ARCHITECTURE.md §4.
        """
        # Hard deterministic check for synthetic URLs
        has_url = bool(re.search(r'https?://\S+|t\.co/\S+', drafted_reply, re.IGNORECASE))
        char_len = len(drafted_reply)

        user_content = f"""Customer Inquiry:
"{customer_text}"

Predicted Intent: {predicted_intent}
Decision: {decision}

Grounding Evidence:
"{evidence_used}"

Drafted Reply:
"{drafted_reply}"

Character Count: {char_len}
Synthetic URL Detected by regex: {has_url}

Evaluate this reply and return ONLY the JSON object."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                max_tokens=750
            )
            raw_text = resp.choices[0].message.content.strip()

            # Parse JSON from response
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(raw_text)

            groundedness = int(parsed.get("groundedness_score", 3))
            tone = int(parsed.get("tone_score", 3))
            decision_val = str(parsed.get("judge_decision", "fail")).lower().strip()
            notes = str(parsed.get("judge_notes", ""))

        except Exception as e:
            # Fallback parsing on error
            groundedness = 3
            tone = 3
            decision_val = "fail"
            notes = f"Judge parsing exception: {str(e)}"

        # Deterministic guardrail: synthetic URL forces groundedness=1 and fail
        if has_url and "t.co" not in evidence_used and "http" not in evidence_used:
            groundedness = 1
            decision_val = "fail"
            notes = f"Deterministic Guardrail Triggered: Synthetic URL detected in reply. {notes}"

        # Deterministic guardrail: length > 280 forces fail
        if char_len > 280 and decision_val == "pass":
            decision_val = "fail"
            notes = f"Exceeds Twitter character limit ({char_len} chars > 280). {notes}"

        return {
            "id": row_id,
            "groundedness_score": groundedness,
            "tone_score": tone,
            "judge_decision": decision_val,
            "judge_notes": notes,
            "difficulty_tier": difficulty_tier,
            "char_count": char_len,
            "has_synthetic_url": has_url,
            "judge_model": self.model_name
        }

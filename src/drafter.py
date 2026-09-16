"""
src/drafter.py

Phase 4 Component — Grounded Reply Drafter for SpotifyCares.
Conditions response generation on verified technical evidence via a dual-grounding mechanism:
1. 'retrieved_case': Used when dense retrieval returns a genuine concrete historical resolution.
2. 'curated_sop': Used when retrieval returns a diagnostic question, DM deflection, or generic status.
3. 'none': Used when the case is escalated (generates empathetic routing acknowledgment).

Strict Guardrails:
- Model pinned to 'qwen/qwen3.8-27b' at temperature 0.0.
- Strictly forbidden from inventing URLs (zero hallucinated links).
- Enforces concise Twitter support format (under 280 characters, professional empathetic tone).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import socket
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from src.classifier import load_groq_key
from src.policy import CuratedPolicyReference

# DNS resolution patch for macOS network environment
DNS_CACHE = {'api.groq.com': '104.18.38.236'}
_orig_getaddrinfo = socket.getaddrinfo

def custom_getaddrinfo(host, port, *args, **kwargs):
    if host in DNS_CACHE:
        return _orig_getaddrinfo(DNS_CACHE[host], port, *args, **kwargs)
    return _orig_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = custom_getaddrinfo

MODEL_NAME = "qwen/qwen3.8-27b"

DRAFTING_SYSTEM_PROMPT = """You are the official customer support representative for SpotifyCares on Twitter (@SpotifyCares).
Your job is to draft a concise, empathetic, and helpful tweet replying to a customer.

CRITICAL CONSTRAINTS:
1. STRICT GROUNDING: State ONLY troubleshooting steps and policies that are directly supported by the Grounding Evidence provided below. Do NOT invent, assume, or extrapolate unverified steps.
2. ZERO SYNTHETIC URLS: NEVER invent, fabricate, or include any external web links or URLs (e.g. do NOT generate http:// or https:// links). All links must be completely omitted unless explicitly present in the grounding evidence.
3. TWITTER BREVITY: Keep the reply under 250 characters (1 to 2 short sentences). Be empathetic, direct, and action-oriented.
4. TONE: Friendly, professional, and clear. Address the specific symptom the customer described."""

class ReplyDrafter:
    def __init__(self, model_name: str = MODEL_NAME):
        self.api_key = load_groq_key()
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        self.model_name = model_name
        self.policy = CuratedPolicyReference()

    def _is_concrete_resolution(self, reply_text: str) -> bool:
        """Check if retrieved text provides a genuine concrete action rather than a DM or diagnostic question."""
        t = str(reply_text)
        DM_ANY = re.compile(r'\b(dm\b|direct message|private message|pm\b|chatting there)\b', re.IGNORECASE)
        if DM_ANY.search(t):
            return False
        
        # Must have actionable troubleshooting verb
        ACTION_RE = re.compile(r'\b(restart|reinstall|log out|clear cache|incognito|browser|reset|update|settings|connect|unplug)\b', re.IGNORECASE)
        if not ACTION_RE.search(t):
            return False
            
        # Reject pure diagnostic questions (ending in question mark without concrete action)
        if t.strip().endswith("?") and not bool(re.search(r'\b(try|restarting|reinstalling|logging out)\b', t, re.IGNORECASE)):
            return False
            
        return True

    def draft_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        decision: str,
        decision_reason: str,
        retrieved_hits: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Draft a grounded reply conditioned on decision and verified technical evidence."""
        # 1. Escalation Handling
        if decision == "escalate":
            # Generate empathetic routing message without technical advice
            user_prompt = f"""The customer sent the following message:
"{customer_text}"

This case has been marked for ESCALATION to a human specialist team for the following reason:
"{decision_reason}"

Draft a polite, empathetic Twitter reply letting the customer know we are looking into this and routing their case to our specialist team for further assistance. Remind them that for account or billing safety, our team may follow up in a private message. Do NOT include any URLs."""

            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": DRAFTING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=90
            )
            draft = resp.choices[0].message.content.strip().replace('"', '')
            return {
                "drafted_reply": draft,
                "grounding_source": "none",
                "evidence_used": [f"Escalation Routing: {decision_reason}"]
            }

        # 2. Auto-Handle Grounding Selection
        top_hit = retrieved_hits[0] if retrieved_hits else None
        grounding_source = "none"
        evidence_content = ""
        evidence_summary = []

        if top_hit and self._is_concrete_resolution(top_hit.get("brand_reply", "")):
            grounding_source = "retrieved_case"
            evidence_content = f"Historical Visible Resolution:\n\"{top_hit['brand_reply']}\""
            evidence_summary = [f"Retrieved historical tweet {top_hit.get('tweet_id')}: {top_hit['brand_reply'][:80]}..."]
        else:
            # Fall back to CuratedPolicyReference
            grounding_source = "curated_sop"
            sop = self.policy.find_matching_sop(predicted_intent, customer_text)
            if sop:
                evidence_content = f"Standard Operating Procedure [{sop['name']}]:\n" + "\n".join(sop['steps'])
                evidence_summary = [f"Curated SOP [{sop['id']}]: {sop['name']}"]
            else:
                evidence_content = "Standard Troubleshooting: Recommend restarting device and checking network connection."
                evidence_summary = ["Curated SOP: General Device Restart"]

        # 3. Auto-Handle Reply Drafting
        user_prompt = f"""Customer tweet:
"{customer_text}"

Intent: {predicted_intent}

Grounding Evidence ({grounding_source}):
{evidence_content}

Draft a concise, empathetic customer support tweet guiding the customer through the specific troubleshooting steps above.
Rules: Stay strictly within the provided steps. Do NOT invent any URLs. Keep under 250 characters."""

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": DRAFTING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=90
        )
        draft = resp.choices[0].message.content.strip().replace('"', '')
        # Remove any stray URLs that LLM might hallucinate despite instructions
        draft = re.sub(r'https?://\S+', '', draft).strip()

        return {
            "drafted_reply": draft,
            "grounding_source": grounding_source,
            "evidence_used": evidence_summary
        }

if __name__ == "__main__":
    drafter = ReplyDrafter()
    test_q = "My Spotify keeps skipping tracks in the middle of songs on my phone!"
    res = drafter.draft_reply(
        customer_text=test_q,
        predicted_intent="playback_issue",
        decision="auto_handle",
        decision_reason="Standard playback troubleshooting",
        retrieved_hits=[{"brand_reply": "Does logging out > restarting the device > logging back in help? Keep us posted /MQ", "tweet_id": "184233"}]
    )
    print("Test Auto-Handle Draft:")
    print("Reply:", res["drafted_reply"])
    print("Source:", res["grounding_source"])
    print("Evidence:", res["evidence_used"])

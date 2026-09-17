"""
src/resolution_check.py

Pre-drafting LLM Problem-Resolution Verification Gate.
Targets Failure Mode 2 (40 cases of colloquial phrasing incorrectly escalated)
and Failure Mode 3 (3 cases of unhelpful procedural brush-offs wrongly auto-handled).

Uses an independent evaluation model (openai/gpt-oss-120b via Groq API) to prevent
same-model correlated bias with the drafter/classifier (qwen/qwen3.8-27b).
"""

import os
import re
import json
import time
import hashlib
import socket
from typing import Dict, Any, Optional
from openai import OpenAI

# Robust DNS resolution patch for macOS network environment
DNS_CACHE = {'api.groq.com': '104.18.38.236'}
_orig_getaddrinfo = socket.getaddrinfo

def custom_getaddrinfo(host, port, *args, **kwargs):
    if host in DNS_CACHE:
        return _orig_getaddrinfo(DNS_CACHE[host], port, *args, **kwargs)
    return _orig_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = custom_getaddrinfo

ENV_PATH = "/Users/kavya/Desktop/Groundcheck/.env"
CACHE_PATH = "/Users/kavya/Desktop/Groundcheck/src/resolution_check_cache.json"

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

RESOLUTION_CHECK_SYSTEM_PROMPT = """You are an expert technical QA auditor for Spotify customer support.
Your task is to determine whether a retrieved troubleshooting procedure or solution ACTUALLY resolves the specific issue reported by the customer, or whether it is an irrelevant brush-off that fails to solve the underlying problem.

Evaluation Guidelines:
1. resolves_problem = TRUE if:
   - The customer is describing a client-side glitch, playback disruption, cache corruption, device connectivity issue, or local app failure, AND the candidate troubleshooting steps (e.g. restart, clean reinstall, cache clear, offline toggle, Connect re-pairing) are standard, plausible fixes for that exact technical symptom.
2. resolves_problem = FALSE if:
   - The issue is a backend bug, algorithmic behavior (e.g. shuffle algorithm repetition), missing feature, platform compatibility limitation (e.g. iPhone X display), service outage, billing dispute, or account security issue that CANNOT be fixed by local client troubleshooting.
   - The candidate reply is a generic brush-off (e.g. telling the customer to reinstall when their issue is with recommendation algorithms, licensing, or payment).
   - The customer explicitly states they already tried these steps.

Output Format:
Return ONLY a valid JSON object:
{
  "resolves_problem": true or false,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one concise sentence explaining why the candidate procedure resolves or fails to resolve the customer problem>"
}"""

class ProblemResolutionChecker:
    def __init__(self, model_name: str = "openai/gpt-oss-120b", cache_path: str = CACHE_PATH):
        self.model_name = model_name
        self.cache_path = cache_path
        self.api_key = load_groq_key()
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to persist resolution check cache: {e}")

    def _cache_key(self, customer_text: str, candidate_procedure: str) -> str:
        raw = f"{customer_text.strip()}|||{candidate_procedure.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def check_resolution(
        self,
        customer_text: str,
        candidate_procedure: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Check if the candidate procedure plausibly resolves the customer's problem.
        Returns {"resolves_problem": bool, "confidence": float, "reason": str}.
        """
        if not customer_text or not candidate_procedure:
            return {
                "resolves_problem": False,
                "confidence": 1.0,
                "reason": "Empty customer query or candidate procedure."
            }

        key = self._cache_key(customer_text, candidate_procedure)
        if key in self.cache:
            return self.cache[key]

        prompt = f"""Customer Inquiry:
"{customer_text.strip()}"

Candidate Procedure / Reply:
"{candidate_procedure.strip()}"

Does this candidate procedure actually resolve the customer's specific problem? Return ONLY JSON."""

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": RESOLUTION_CHECK_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=750
                )
                raw = resp.choices[0].message.content.strip()
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    result = {
                        "resolves_problem": bool(parsed.get("resolves_problem", False)),
                        "confidence": float(parsed.get("confidence", 0.8)),
                        "reason": str(parsed.get("reason", "")).strip()
                    }
                else:
                    result = {
                        "resolves_problem": False,
                        "confidence": 0.5,
                        "reason": "Failed to parse JSON from resolution checker response."
                    }

                self.cache[key] = result
                self._save_cache()
                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    # Safe fallback: escalate if checker fails
                    return {
                        "resolves_problem": False,
                        "confidence": 0.0,
                        "reason": f"Resolution checker API error: {str(e)}"
                    }

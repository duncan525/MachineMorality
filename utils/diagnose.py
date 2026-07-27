"""
diagnose_logprobs.py

Run this ON YOUR MACHINE (where Ollama is running) to see the REAL error that
the test harness is swallowing. No try/except -- if something throws, you get
the full traceback.

    python diagnose_logprobs.py
"""

import json
import logprob_disagreement as L

MODEL = "gemma4"

# --- 1. raw call: exactly what get_agent_opinion sends -----------------------
import requests

prompt = "Pick the best option. Respond with exactly one letter: A, B, C, or D.\nA) x\nB) y\nC) z\nD) w"
resp = requests.post(
    L.DEFAULT_OLLAMA_URL,
    json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 20,
        "options": {"num_predict": 1, "temperature": 0},
    },
    timeout=60,
)
print("HTTP status:", resp.status_code)
data = resp.json()
print("Top-level keys returned:", list(data.keys()))
print("logprobs field type:", type(data.get("logprobs")))
print("\nFull logprobs payload:")
print(json.dumps(data.get("logprobs"), indent=2)[:2000])

# --- 2. now call the actual function, no safety net --------------------------
print("\n--- calling get_agent_opinion directly ---")
op = L.get_agent_opinion(
    schema="You reason using the care ethics lens.",
    lens_name="care ethics",
    user_prompt="Please evaluate how important family is in life.",
    aspect="beliefs",
    candidates=["belief text A", "belief text B", "belief text C", "belief text D"],
    model=MODEL,
)
print("opinion distribution:", op)
# hf_ping.py
import os, time, requests, sys, tomllib  # Py 3.11+

def get_secret(key: str) -> str | None:
    # env first, then .streamlit/secrets.toml
    if v := os.environ.get(key):
        return v
    try:
        with open(".streamlit/secrets.toml", "rb") as f:
            return tomllib.load(f).get(key)
    except FileNotFoundError:
        return None

TOKEN = get_secret("HF_TOKEN")
if not TOKEN:
    raise SystemExit("HF_TOKEN not found (set env or .streamlit/secrets.toml).")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-chat-hf"
URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HDR = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Llama-2 chat models like the [INST] format; harmless for others too.
payload = {
    "inputs": "<s>[INST] ping [/INST]>",
    "parameters": {"max_new_tokens": 8, "temperature": 0},
    "options": {"wait_for_model": True},  # handles cold starts
}

t0 = time.time()
r = requests.post(URL, headers=HDR, json=payload, timeout=120)
print("Model:", MODEL)
print("Status:", r.status_code, f"({time.time()-t0:.2f}s)")
print(r.text[:800])

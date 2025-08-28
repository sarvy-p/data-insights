# hf_model_access_check.py
import os, sys, time
import tomllib  # Py ≥3.11
from huggingface_hub import HfApi, hf_hub_download, InferenceClient
from huggingface_hub.utils import HfHubHTTPError

def get_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        with open(".streamlit/secrets.toml", "rb") as f:
            return tomllib.load(f).get("HF_TOKEN")
    except FileNotFoundError:
        return None

def main():
    token = get_token()
    if not token:
        raise SystemExit("❌ HF_TOKEN not found (env or .streamlit/secrets.toml).")

    repo = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-chat-hf"
    api = HfApi()

    # 1) Metadata (sanity)
    try:
        info = api.model_info(repo_id=repo, token=token)
        print(f"ℹ️ model_info: gated={getattr(info, 'gated', None)}, disabled={getattr(info, 'disabled', None)}")
    except Exception as e:
        print(f"⚠️ model_info error: {e}")

    # 2) File access (definitive license check)
    try:
        p = hf_hub_download(repo_id=repo, filename="config.json", token=token, local_dir=".hf_check", local_dir_use_symlinks=False)
        print(f"✅ File access OK: downloaded {p}")
        file_ok = True
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", None)
        print(f"❌ File access error ({code}): {e}")
        file_ok = False

    # 3) Inference (serverless) — optional but nice to confirm
    try:
        client = InferenceClient(model=repo, token=token, provider="hf-inference")
        t0 = time.time()
        out = client.text_generation("<s>[INST] ping [/INST]>", max_new_tokens=8, temperature=0, return_full_text=False)
        print(f"✅ Inference OK ({time.time()-t0:.2f}s): {str(out)[:120]}")
        infer_ok = True
    except Exception as e:
        # 403 => license not accepted; 503 => cold start (not an access issue)
        print(f"⚠️ Inference error: {e}")
        infer_ok = False

    if file_ok or infer_ok:
        print("🎉 Access CONFIRMED for:", repo)
    else:
        print("🚫 Access NOT confirmed. If you just accepted the license, wait 2–3 minutes and re-run.")

if __name__ == "__main__":
    main()

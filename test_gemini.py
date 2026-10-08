import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing.")

url = "https://generativelanguage.googleapis.com/v1beta/interactions"

print("Sending direct HTTPS request...", flush=True)

try:
    response = requests.post(
        url,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "model": "gemini-3.8-flash",
            "input": "Reply with exactly: Gemini works!",
        },
        timeout=(5, 20),
    )

    print("HTTP status:", response.status_code, flush=True)
    print("Response:", response.text[:3000], flush=True)

except requests.exceptions.Timeout:
    print("FAILED: Request timed out.", flush=True)

except requests.exceptions.RequestException as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)

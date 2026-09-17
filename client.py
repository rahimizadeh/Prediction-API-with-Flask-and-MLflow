"""Simple client for the salary prediction API."""

import requests

payload = {"experience": 4, "age": 30, "interview_score": 8}

try:
    response = requests.post(
        "http://localhost:5001/predict",
        json=payload,
        timeout=8,
    )
    response.raise_for_status()
    print(response.json())
except requests.exceptions.RequestException as exc:
    print(f"Request failed: {exc}")

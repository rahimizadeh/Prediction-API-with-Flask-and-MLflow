# Client side. 
# Test the API.
# Run this code in another terminal: python client.py

import requests

try:
    response = requests.post(
        "http://localhost:5001/predict",
        json={"level": 6.5},
        timeout= 8  # Fail fast if no connection
    )
    print(response.json())
except requests.exceptions.ConnectionError:
    print("Server not running!")




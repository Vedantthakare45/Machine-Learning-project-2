"""Send an image to the running API as base64 and print the JSON response.

Usage: python test_client.py path/to/digit.png [http://localhost:5000]
"""
import base64
import sys

import requests

if len(sys.argv) < 2:
    sys.exit("Usage: python test_client.py path/to/digit.png [base_url]")

url = (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5000") + "/predict"
with open(sys.argv[1], "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

resp = requests.post(url, json={"image": encoded}, timeout=30)
print(resp.status_code, resp.json())

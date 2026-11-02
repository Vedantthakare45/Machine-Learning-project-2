"""Step 3: Test the endpoint (works against a running server or in-process)."""
import json
import sys

import app as flask_app


def check(client, path, expected_status=200):
    resp = client.get(path)
    assert resp.status_code == expected_status, f"{path} -> {resp.status_code}"
    return resp.get_json()


def main() -> None:
    client = flask_app.app.test_client()

    print("GET /health")
    print(json.dumps(check(client, "/health"), indent=2))

    print("\nGET /forecast")
    data = check(client, "/forecast")
    print(json.dumps(data, indent=2))
    assert len(data["forecast"]) == 12
    assert data["forecast"][0]["month"] == "1961-01"
    assert all(r["lower_95"] < r["forecast"] < r["upper_95"] for r in data["forecast"])

    assert len(check(client, "/forecast?periods=24")["forecast"]) == 24
    check(client, "/forecast?periods=0", 400)
    check(client, "/forecast?periods=abc", 400)

    print("\nAll tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

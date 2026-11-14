"""Send a steady stream of requests to the Iris service so the dashboard has data.

Usage:
  python load_generator.py                      # ~10 req/s, ~5% invalid requests (HTTP 400)
  python load_generator.py --bad-rate 0.3       # 30% invalid -> triggers the HighErrorRate alert
  python load_generator.py --url http://localhost:5000 --rps 20 --duration 120

Standard library only (no extra installs).
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request


def send(url, body):
    req = urllib.request.Request(
        url + "/predict",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except urllib.error.URLError:
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:5000")
    p.add_argument("--rps", type=float, default=10)
    p.add_argument("--bad-rate", type=float, default=0.05, help="fraction of invalid requests")
    p.add_argument("--duration", type=float, default=0, help="seconds (0 = run until Ctrl+C)")
    args = p.parse_args()

    counts = {}
    start = time.time()
    try:
        while not args.duration or time.time() - start < args.duration:
            if random.random() < args.bad_rate:
                body = {"features": "oops"}
            else:
                body = {"features": [round(random.uniform(4.3, 7.9), 1), round(random.uniform(2.0, 4.4), 1),
                                     round(random.uniform(1.0, 6.9), 1), round(random.uniform(0.1, 2.5), 1)]}
            code = send(args.url, body)
            counts[code] = counts.get(code, 0) + 1
            time.sleep(1 / args.rps)
    except KeyboardInterrupt:
        pass
    print("Status counts:", counts)


if __name__ == "__main__":
    main()

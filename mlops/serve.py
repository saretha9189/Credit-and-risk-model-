"""A small scoring service (standard library only, so it runs anywhere).

    python -m mlops.serve            # http://localhost:8080
    curl -s localhost:8080/health
    curl -s -X POST localhost:8080/score -d @example_request.json

Every scored request is appended to artifacts/prediction_log.jsonl so the monitor can
compare live traffic with the training data.
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib
import numpy as np

from src.config import ARTIFACT_DIR
from mlops.schema import ValidationError, clean_records

ARTIFACTS = ARTIFACT_DIR


class Scorer:
    def __init__(self, path=ARTIFACTS / "model_bundle.joblib", log_path=ARTIFACTS / "prediction_log.jsonl"):
        self.b = joblib.load(path)
        self.log_path = log_path

    def score(self, records: list) -> dict:
        raw = clean_records(records, self.b["raw_cols"])
        p = self.b["model"].predict_proba(self.b["fb"].transform(raw))[:, 1]
        out = [{"probability_of_default": round(float(x), 4),
                "decision": "DECLINE" if x > self.b["cutoff"] else "APPROVE"} for x in p]
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "model_version": self.b["version"],
                                "scores": [float(x) for x in p]}) + "\n")
        return {"model_version": self.b["version"], "cutoff": self.b["cutoff"], "results": out,
                "ignored_fields": raw.attrs.get("unknown_fields", [])}


def make_handler(scorer: Scorer):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model_version": scorer.b["version"]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                records = body["records"] if isinstance(body, dict) else body
                self._send(200, scorer.score(records))
            except (ValidationError, KeyError, json.JSONDecodeError) as e:
                self._send(422, {"error": str(e)})

        def log_message(self, *a):
            pass
    return H


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", 8080), make_handler(Scorer()))
    print("scoring service on http://localhost:8080  (POST /score, GET /health)")
    srv.serve_forever()

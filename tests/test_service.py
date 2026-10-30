"""End-to-end test of the scoring service. Needs `python -m mlops.train_and_register` first."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest

from mlops.serve import ARTIFACTS, Scorer, make_handler

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "model_bundle.joblib").exists(), reason="model not trained yet")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    scorer = Scorer(log_path=tmp_path_factory.mktemp("logs") / "log.jsonl")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(scorer))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", scorer
    srv.shutdown()


def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_health(server):
    code, body = call(server[0] + "/health")
    assert code == 200 and body["status"] == "ok"


def test_scores_real_applicants_and_matches_offline_model(server):
    url, scorer = server
    raw = pd.read_pickle(ARTIFACTS / "holdout.pkl").drop(columns=["TARGET"]).head(5)
    recs = json.loads(raw.to_json(orient="records"))
    code, body = call(url + "/score", {"records": recs})
    assert code == 200 and len(body["results"]) == 5
    assert all(0 <= r["probability_of_default"] <= 1 for r in body["results"])
    offline = scorer.b["model"].predict_proba(scorer.b["fb"].transform(raw))[:, 1]
    assert [r["probability_of_default"] for r in body["results"]] == pytest.approx(list(offline), abs=1e-3)


def test_bad_input_gets_422_not_a_score(server):
    code, body = call(server[0] + "/score", {"records": [{"AGE": 3}]})
    assert code == 422 and "error" in body

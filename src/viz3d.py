"""Builds the 3D dashboard page (Plotly.js, animated, pink/purple)."""
import json

from .config import TEMPLATE

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def fragment(payload: dict) -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("__DATA__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))


def write_dashboard(payload: dict, out_dir):
    frag = fragment(payload)
    (out_dir / "dashboard_fragment.html").write_text(frag, encoding="utf-8")
    standalone = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                  "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"></head><body>"
                  + frag + "</body></html>")
    (out_dir / "dashboard.html").write_text(standalone, encoding="utf-8")

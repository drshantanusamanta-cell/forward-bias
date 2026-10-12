"""
Headline sentiment from Google News RSS, scored with VADER (a light, general-purpose model).
Display only — not part of the matrix score.
© 2026 Dr Shantanu Samanta. Educational and research use only.
"""
from __future__ import annotations
import re, datetime as dt
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import requests

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:  # package missing
    _VADER = None


def headlines(company: str, days: int = 14, limit: int = 25) -> list[dict]:
    name = re.sub(r"\b(limited|ltd\.?|the)\b", "", company, flags=re.I).strip()
    url = ("https://news.google.com/rss/search?q=" + quote_plus(f'"{name}" when:{days}d')
           + "&hl=en-IN&gl=IN&ceid=IN:en")
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
    except Exception:
        return []
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src = it.find("source")
        try:
            when = parsedate_to_datetime(it.findtext("pubDate")).date()
        except Exception:
            when = None
        score = _VADER.polarity_scores(title)["compound"] if _VADER else None
        out.append({"date": when, "title": title, "source": src.text if src is not None else "",
                    "link": it.findtext("link"), "sentiment": score})
        if len(out) >= limit:
            break
    return out


def summary(items: list[dict]) -> dict:
    s = [x["sentiment"] for x in items if x.get("sentiment") is not None]
    if not s:
        return {"n": len(items), "avg": None, "pos": 0, "neg": 0, "label": "No data"}
    avg = sum(s) / len(s)
    pos = sum(1 for v in s if v >= 0.2); neg = sum(1 for v in s if v <= -0.2)
    label = "Positive" if avg >= 0.15 else "Negative" if avg <= -0.15 else "Mixed / neutral"
    return {"n": len(s), "avg": avg, "pos": pos, "neg": neg, "label": label}

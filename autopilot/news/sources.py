"""Free news sources. Every source is best-effort: failures are logged, never fatal."""
from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36", "Accept-Language": "en-IN,en;q=0.9"}


def item_id(source: str, symbol: str, title: str, url: str) -> str:
    return hashlib.sha1(f"{source}|{symbol}|{title.strip().lower()}|{url}".encode()).hexdigest()[:16]


def parse_rss(xml_text: str, symbol: str, source: str = "google_news") -> list[dict]:
    items = []
    root = ET.fromstring(xml_text)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        url = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate")
        try:
            published = parsedate_to_datetime(pub).astimezone(timezone.utc) if pub else None
        except (TypeError, ValueError):
            published = None
        publisher = (it.findtext("source") or "").strip()
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3]
        items.append({"id": item_id(source, symbol, title, url), "symbol": symbol, "source": source,
                      "publisher": publisher, "title": title, "url": url, "published": published,
                      "official": False})
    return items


class GoogleNews:
    URL = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=20, headers=UA, follow_redirects=True)

    def fetch(self, symbol: str, name: str = "") -> list[dict]:
        who = f'"{name}"' if name else f'"{symbol}" NSE'
        q = quote_plus(f"{who} when:2d")
        try:
            r = self.http.get(self.URL.format(q=q))
            r.raise_for_status()
            return parse_rss(r.text, symbol)
        except Exception as exc:
            log.warning("google news failed for %s: %s", symbol, exc)
            return []


class NSEAnnouncements:
    """Official exchange filings. NSE's website API is unofficial and bot-hostile:
    it needs a cookie from the homepage and may still block. [TBD verify fields]"""

    HOME = "https://www.nseindia.com"
    API = "https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={sym}"

    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=20, headers={**UA, "Referer": self.HOME},
                                         follow_redirects=True)
        self._warm = False

    def fetch(self, symbol: str, name: str = "") -> list[dict]:
        try:
            if not self._warm:
                self.http.get(self.HOME)
                self._warm = True
            r = self.http.get(self.API.format(sym=quote_plus(symbol)))
            r.raise_for_status()
            rows = r.json()
            rows = rows.get("data", rows) if isinstance(rows, dict) else rows
        except Exception as exc:
            log.warning("NSE announcements failed for %s: %s", symbol, exc)
            self._warm = False
            return []
        out = []
        for a in rows[:20]:
            title = " - ".join(x for x in (a.get("desc"), a.get("attchmntText")) if x)[:400]
            try:   # NSE timestamps are IST
                published = datetime.strptime(a.get("an_dt", ""), "%d-%b-%Y %H:%M:%S") \
                    .replace(tzinfo=IST).astimezone(timezone.utc)
            except ValueError:
                published = None
            url = a.get("attchmntFile") or self.HOME
            out.append({"id": item_id("nse", symbol, title, url), "symbol": symbol, "source": "nse",
                        "publisher": "NSE filing", "title": title, "url": url, "published": published,
                        "official": True})
        return out

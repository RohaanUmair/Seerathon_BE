"""
corpus.py — Live corpus search functions.
These hit the IslamicDesk Seerathon API in real-time.
No database, no vector store, no caching.
"""

import json
import re
import httpx

BASE_URL = "https://api.islamicdesk.com/api/seerathon/corpus"
TIMEOUT = 15.0

STOPWORDS = {
    'what', 'did', 'the', 'prophet', 'pbuh', 'ﷺ', 'how', 'was', 'is', 'are', 
    'were', 'a', 'an', 'of', 'in', 'on', 'tell', 'me', 'about', 'who', 'can', 
    'you', 'describe', 'give', 'information', 'details', 'regarding'
}

def extract_keyword(query: str) -> str:
    """Extract clean search terms from natural language questions."""
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    words = [w.lower() for w in cleaned.split()]
    if any(w in words for w in ['look', 'looks', 'physique', 'height', 'stature', 'face', 'eyes', 'hair', 'feature', 'features', 'body', 'physical', 'complexion']):
        return 'appearance'
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join(keywords) if keywords else query


async def search_shamail(query: str) -> str:
    """
    Search the Shamail corpus for entries related to the query.
    Returns JSON string of results for the agent to consume.
    """
    search_term = extract_keyword(query)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.get(
                f"{BASE_URL}/shamail",
                params={"q": search_term, "limit": 5, "include_hikayat": "true"},
            )
            response.raise_for_status()
            data = response.json()

            # API returns { error: false, data: { items: [...] } }
            items = []
            raw = data.get("data", data)
            if isinstance(raw, dict):
                items = raw.get("items", [])
            elif isinstance(raw, list):
                items = raw

            if not items:
                return "[]"

            entries = []
            for item in items:
                en = item.get("en", {})
                entry = {
                    "id": item.get("id", ""),
                    "title": en.get("title", item.get("title", "")),
                    "text": en.get("hadeesTarjama", item.get("text", "")),
                    "hawala": en.get("hadeesHawala", ""),
                    "points": en.get("points", []),
                    "hikayat": en.get("hikayat", item.get("hikayat", "")),
                    "category": (item.get("category", {}).get("name", {}).get("en", "")),
                }
                entries.append(entry)

            return json.dumps(entries, ensure_ascii=False)
        except Exception as e:
            return f"Error searching shamail: {str(e)}"


async def search_timeline(query: str) -> str:
    """
    Search the Seerah Timeline corpus for entries related to the query.
    Returns JSON string of results for the agent to consume.
    """
    search_term = extract_keyword(query)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.get(
                f"{BASE_URL}/timeline",
                params={"q": search_term, "limit": 5},
            )
            response.raise_for_status()
            data = response.json()

            items = []
            raw = data.get("data", data)
            if isinstance(raw, dict):
                items = raw.get("items", [])
            elif isinstance(raw, list):
                items = raw

            if not items:
                return "[]"

            entries = []
            for item in items:
                en = item.get("en", {})
                entry = {
                    "id": item.get("id", ""),
                    "title": en.get("title", item.get("title", "")),
                    "text": en.get("description", en.get("text", item.get("text", ""))),
                    "year": item.get("year", ""),
                    "category": (item.get("category", {}).get("name", {}).get("en", "")),
                }
                entries.append(entry)

            return json.dumps(entries, ensure_ascii=False)
        except Exception as e:
            return f"Error searching timeline: {str(e)}"

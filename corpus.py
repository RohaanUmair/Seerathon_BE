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


async def _fetch_corpus_items(client: httpx.AsyncClient, endpoint: str, term: str, extra_params: dict = None) -> list:
    """Helper to fetch items from corpus endpoint with a given term."""
    params = {"q": term, "limit": 5}
    if extra_params:
        params.update(extra_params)
    try:
        response = await client.get(f"{BASE_URL}/{endpoint}", params=params)
        if response.status_code == 200:
            data = response.json()
            raw = data.get("data", data)
            if isinstance(raw, dict):
                return raw.get("items", [])
            elif isinstance(raw, list):
                return raw
    except Exception:
        pass
    return []


async def search_shamail(query: str) -> str:
    """
    Search the Shamail corpus for entries related to the query.
    Returns JSON string of results for the agent to consume.
    """
    search_term = extract_keyword(query)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            items = await _fetch_corpus_items(client, "shamail", search_term, {"include_hikayat": "true"})
            
            # If multi-word search returned 0 items, retry with individual words
            if not items and " " in search_term:
                words = sorted([w for w in search_term.split() if len(w) > 2], key=len, reverse=True)
                for word in words:
                    items = await _fetch_corpus_items(client, "shamail", word, {"include_hikayat": "true"})
                    if items:
                        break

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
            items = await _fetch_corpus_items(client, "timeline", search_term)
            
            # If multi-word search returned 0 items, retry with individual words
            if not items and " " in search_term:
                words = sorted([w for w in search_term.split() if len(w) > 2], key=len, reverse=True)
                for word in words:
                    items = await _fetch_corpus_items(client, "timeline", word)
                    if items:
                        break

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

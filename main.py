"""
main.py — FastAPI entry point with input guardrail and /chat endpoint.
"""

import re
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import Runner
from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

from agent import seerah_agent, seerah_agent_groq

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seerah")

# ── App setup ──────────────────────────────────────────────────────────
app = FastAPI(title="Seerah Q&A Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Seerah Backend is running"}

# ── Request / Response models ──────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    citation: str | None = None
    type: str  # "answer" | "fatwa_redirect" | "fallback"


# ── Fatwa Guardrail Response ──────────────────────────────────────────
FATWA_RESPONSE = (
    "I'm not able to provide religious rulings or fatwas. "
    "Please consult a qualified Islamic scholar (alim) for guidance."
)

FALLBACK_PHRASE = "This topic isn't covered in my knowledge base"


def parse_agent_output(raw: str) -> ChatResponse:
    """Parse the agent's raw text into structured answer + citation + type."""
    citation = None
    answer = raw

    # Extract SOURCE citation lines
    source_lines = []
    remaining_lines = []
    for line in raw.split("\n"):
        if line.strip().startswith("SOURCE:"):
            source_lines.append(line.strip())
        else:
            remaining_lines.append(line)

    if source_lines:
        valid_sources = [
            l for l in source_lines
            if not any(bad in l.lower() for bad in ["source: none", "none — none", "source: n/a", "source: null", "none"])
        ]
        if valid_sources:
            citation = "\n".join(valid_sources)
        answer = "\n".join(remaining_lines).strip()

    # Determine response type
    if FALLBACK_PHRASE.lower() in raw.lower() or "high demand" in raw.lower() or "encountered an error" in raw.lower():
        resp_type = "fallback"
    elif any(kw in raw.lower() for kw in ["not able to provide religious rulings", "consult a qualified islamic scholar"]):
        resp_type = "fatwa_redirect"
    else:
        resp_type = "answer"

    return ChatResponse(answer=answer, citation=citation, type=resp_type)


# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "service": "Seerah Q&A Chatbot"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint — runs agent with built-in guardrails and Groq fallback."""

    try:
        # Run primary agent (Gemini)
        result = await Runner.run(seerah_agent, req.message)
        raw_output = result.final_output
        logger.info(f"Agent responded via Gemini ({len(raw_output)} chars)")
        
    except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered):
        # Guardrail caught a fatwa request
        return ChatResponse(
            answer=FATWA_RESPONSE,
            citation=None,
            type="fatwa_redirect",
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Gemini agent error: {error_msg[:200]}")

        # Attempt Groq fallback if Gemini encounters rate limit or API error
        if seerah_agent_groq:
            logger.info("Falling back to Groq (Llama 3.3)...")
            try:
                result = await Runner.run(seerah_agent_groq, req.message)
                raw_output = result.final_output
                logger.info(f"Agent responded via Groq fallback ({len(raw_output)} chars)")
            except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered):
                return ChatResponse(
                    answer=FATWA_RESPONSE,
                    citation=None,
                    type="fatwa_redirect",
                )
            except Exception as groq_err:
                logger.error(f"Groq fallback error: {groq_err}")
                raw_output = (
                    "The service is currently experiencing high demand. "
                    "Please wait a moment and try again."
                )
        else:
            raw_output = (
                "The service is currently experiencing high demand. "
                "Please wait a moment and try again."
            )

    # 3. Parse and return
    return parse_agent_output(raw_output)


@app.get("/source/{source_type}/{source_id}")
async def get_source(source_type: str, source_id: str):
    """Fetch full details for a single corpus entry (for modal display)."""
    import httpx

    endpoint_map = {
        "shamail": "shamail",
        "timeline": "timeline",
    }
    endpoint = endpoint_map.get(source_type.lower())
    if not endpoint:
        return {"error": True, "message": "Invalid source type"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            item = None
            url = f"https://api.islamicdesk.com/api/seerathon/corpus/{endpoint}/{source_id}"
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data.get("data"), dict):
                    item = data.get("data", {}).get("item")

            # Fallback: if direct lookup by ID fails (e.g. source_id is #1 or title), search corpus with query string
            if not item:
                search_url = f"https://api.islamicdesk.com/api/seerathon/corpus/{endpoint}"
                search_resp = await client.get(search_url, params={"q": source_id, "limit": 1})
                if search_resp.status_code == 200:
                    search_data = search_resp.json()
                    items = search_data.get("data", {}).get("items", [])
                    if items:
                        item = items[0]

            if not item:
                return {"error": True, "message": "Source item not found"}

            en = item.get("en", {})
            cat = item.get("category", {})
            cat_name = cat.get("name", {}).get("en", "") if isinstance(cat, dict) and isinstance(cat.get("name"), dict) else ""

            return {
                "error": False,
                "data": {
                    "id": item.get("id", source_id),
                    "source": source_type,
                    "title": en.get("title", ""),
                    "text": en.get("hadeesTarjama", en.get("description", "")),
                    "hawala": en.get("hadeesHawala", ""),
                    "points": en.get("points", []),
                    "hikayat": en.get("hikayat", ""),
                    "category": cat_name,
                    "urdu_title": item.get("ur", {}).get("title", "") if isinstance(item.get("ur"), dict) else "",
                    "urdu_text": item.get("ur", {}).get("hadeesTarjama", "") if isinstance(item.get("ur"), dict) else "",
                },
            }
    except Exception as e:
        logger.error(f"Source fetch error: {str(e)[:200]}")
        return {"error": True, "message": "Could not fetch source details"}

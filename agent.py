"""
agent.py — Seerah Q&A Agent definition using OpenAI Agents SDK with Gemini.
Uses LLM-based guardrail agents (not keyword matching) for fatwa detection.
"""

import os
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    set_default_openai_api,
    function_tool,
    InputGuardrail,
    OutputGuardrail,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
)
from corpus import search_shamail, search_timeline

load_dotenv()

# ── Gemini via OpenAI-compatible endpoint ──────────────────────────────
gemini_client = AsyncOpenAI(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")

# ── Groq via OpenAI-compatible endpoint ────────────────────────────────
groq_api_key = os.environ.get("GROQ_API_KEY", "")
groq_client = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
) if groq_api_key else None

groq_model_70b = OpenAIChatCompletionsModel(
    model="llama-3.3-70b-versatile",
    openai_client=groq_client,
) if groq_client else None

groq_model_8b = OpenAIChatCompletionsModel(
    model="llama-3.1-8b-instant",
    openai_client=groq_client,
) if groq_client else None


# ── Tool definitions ───────────────────────────────────────────────────
@function_tool
async def search_shamail_tool(query: str) -> str:
    """Search the Shamail corpus for entries about the Prophet Muhammad ﷺ.
    Use this to find information about the Prophet's appearance, character,
    habits, clothing, eating, drinking, sleeping, worship, and companions' descriptions.
    Args:
        query: The search keyword or topic extracted from the user's question.
    """
    return await search_shamail(query)


@function_tool
async def search_timeline_tool(query: str) -> str:
    """Search the Seerah Timeline corpus for key events in the Prophet's ﷺ life.
    Use this to find information about historical events such as battles, treaties,
    migration (Hijrah), revelation, and other milestones.
    Args:
        query: The search keyword or topic extracted from the user's question.
    """
    return await search_timeline(query)


# ── System prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Seerah Q&A assistant. You answer questions about the Prophet Muhammad ﷺ using ONLY the corpus entries your tools return.

WORKFLOW (follow strictly):
1. Extract the key topic from the user's question
2. Call search_shamail_tool with that topic
3. Call search_timeline_tool with that topic
4. If both return empty results → give safe fallback (step 6)
5. If results found → answer ONLY from those entries, never add outside knowledge
6. Empty results fallback: "This topic isn't covered in my knowledge base. Try asking about the Prophet's ﷺ character, companions, or key Seerah events."

CITATION (mandatory on every answer):
End every answer with exactly this format using the exact "id" value from the tool result:
SOURCE: [Shamail|Timeline] #[exact id] — [title]
Example: SOURCE: Shamail #674ed107f8a58b001f4e554f — The Beloved Prophet's ﷺ stature/height

If you used multiple sources, list each one on its own line.

FATWA REFUSAL (mandatory):
If the question contains any of these words — fatwa, ruling, halal, haram, permissible, forbidden, allowed, sinful, fiqh — do NOT search the corpus. Immediately respond:
"I'm not able to provide religious rulings or fatwas. Please consult a qualified Islamic scholar (alim) for guidance."
"""


# ══════════════════════════════════════════════════════════════════════
# LLM-BASED GUARDRAIL AGENTS
# Instead of keyword matching, we use a small LLM "judge" agent that
# understands INTENT — so even rephrased fatwa questions get caught.
# ══════════════════════════════════════════════════════════════════════


# ── Structured output types for guardrail agents ─────────────────────
class FatwaCheckOutput(BaseModel):
    is_fatwa_request: bool
    reasoning: str


# ── Input Guardrail: checks user question before agent runs ──────────
input_guardrail_agent = Agent(
    name="FatwaInputGuardrail",
    instructions="""You are a content classifier. Your ONLY job is to determine
whether the user's message is asking for a religious ruling (fatwa), legal
Islamic opinion, or judgment on whether something is halal/haram/permissible/
forbidden/sinful.

Flag as TRUE if the message:
- Asks whether something is halal, haram, allowed, forbidden, sinful, or permissible
- Requests a fatwa, ruling, or fiqh opinion
- Asks "is X okay in Islam?" or "can a Muslim do X?" type questions
- Asks about Islamic law or jurisprudence rulings

Flag as FALSE if the message:
- Asks about the Prophet's ﷺ life, character, appearance, or biography (Seerah)
- Asks about historical events, battles, companions, or timeline
- Is a general knowledge question about Islam's history
- Is casual conversation or greetings

Respond ONLY with your structured output. No explanations outside the schema.""",
    output_type=FatwaCheckOutput,
    model=groq_model_70b if groq_model_70b else "gemini-2.5-flash",
)


async def fatwa_input_guardrail_fn(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Run the guardrail agent to classify user input."""
    try:
        result = await Runner.run(input_guardrail_agent, input, context=ctx.context)
        is_fatwa = result.final_output.is_fatwa_request
    except Exception as e:
        print(f"[Guardrail Warning] Input check failed: {e}. Defaulting to safe pass.")
        is_fatwa = False

    return GuardrailFunctionOutput(
        output_info={"is_fatwa": is_fatwa},
        tripwire_triggered=is_fatwa,
    )


fatwa_input_guard = InputGuardrail(
    guardrail_function=fatwa_input_guardrail_fn,
    name="FatwaInputGuardrail",
)


# ── Output Guardrail: checks agent response before returning ────────
output_guardrail_agent = Agent(
    name="FatwaOutputGuardrail",
    instructions="""You are an output safety checker. Your ONLY job is to
determine whether the given text contains a religious ruling, fatwa,
or a definitive statement about something being halal/haram/permissible/
forbidden/sinful in Islamic law.

Flag as TRUE if the text:
- Issues or contains a fatwa or religious ruling
- States definitively that something is halal, haram, allowed, or forbidden
- Provides fiqh/jurisprudence opinions as if authoritative

Flag as FALSE if the text:
- Describes historical events from the Prophet's ﷺ life (Seerah)
- Shares hadith narrations about the Prophet's ﷺ character or appearance
- Contains the word "halal"/"haram" only in a historical/descriptive context
  (e.g. "the Prophet ﷺ ate halal food") without issuing a ruling
- Is a safe fallback or disclaimer message

Respond ONLY with your structured output.""",
    output_type=FatwaCheckOutput,
    model=groq_model_70b if groq_model_70b else "gemini-2.5-flash",
)


async def fatwa_output_guardrail_fn(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output: str,
) -> GuardrailFunctionOutput:
    """Run the guardrail agent to classify agent output."""
    try:
        result = await Runner.run(
            output_guardrail_agent, str(output), context=ctx.context
        )
        is_fatwa = result.final_output.is_fatwa_request
    except Exception as e:
        print(f"[Guardrail Warning] Output check failed: {e}. Defaulting to safe pass.")
        is_fatwa = False

    return GuardrailFunctionOutput(
        output_info={"is_fatwa": is_fatwa},
        tripwire_triggered=is_fatwa,
    )


fatwa_output_guard = OutputGuardrail(
    guardrail_function=fatwa_output_guardrail_fn,
    name="FatwaOutputGuardrail",
)


# ── Main Agent definitions ───────────────────────────────────────────
seerah_agent = Agent(
    name="SeerahQAAgent",
    instructions=SYSTEM_PROMPT,
    model="gemini-2.5-flash",
    tools=[search_shamail_tool, search_timeline_tool],
    input_guardrails=[fatwa_input_guard],
    output_guardrails=[fatwa_output_guard],
)

# ── Groq Fallback Agent ──────────────────────────────────────────────
seerah_agent_groq = Agent(
    name="SeerahQAAgentGroq",
    instructions=SYSTEM_PROMPT,
    model=groq_model_70b if groq_model_70b else "gemini-2.5-flash",
    tools=[search_shamail_tool, search_timeline_tool],
    input_guardrails=[fatwa_input_guard],
    output_guardrails=[fatwa_output_guard],
) if groq_model_70b else None

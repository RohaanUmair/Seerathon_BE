# Seerat Ki Dunya - Backend API

FastAPI backend with OpenAI Agents SDK, Gemini primary model, Groq fallback, and IslamicDesk corpus search integration.

## Setup
1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Set environment variables in `.env`:
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`

3. Run server:
   ```bash
   uvicorn main:app --reload
   ```

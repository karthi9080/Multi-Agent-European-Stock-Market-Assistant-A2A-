# Stock Info Agent (LangGraph + A2A)

A LangGraph ReAct agent exposed over the Agent-to-Agent (A2A) protocol. It answers questions about the main stock market index of 18 European countries using live Yahoo Finance data (`yfinance`).

## Prerequisites

1. **uv**: <https://docs.astral.sh/uv/getting-started/installation/>
2. **Python 3.12+**
3. A Google API key in `.env`: `GOOGLE_API_KEY="your_api_key_here"` (optionally `GEMINI_MODEL="gemini-3.5-flash-lite"`)

## 1. Install dependencies

```bash
uv sync
```

## 2. Run the agent

```bash
uv run app/__main__.py
```

The agent runs on `http://localhost:10004` (override with `STOCK_AGENT_HOST` / `STOCK_AGENT_PORT`). Its Agent Card is served at `http://localhost:10004/.well-known/agent.json`.

## 3. Try it

Run the host agent from `../host_agent_adk` (`uv run adk web`), or send an A2A message yourself with the `a2a-sdk` client.

## 4. Tests

```bash
uv run pytest -m "not live"   # offline, no API key needed
uv run pytest -m live         # real Yahoo Finance, all 18 countries
```

## References
- https://github.com/google/a2a-python
- https://codelabs.developers.google.com/intro-a2a-purchasing-concierge#1

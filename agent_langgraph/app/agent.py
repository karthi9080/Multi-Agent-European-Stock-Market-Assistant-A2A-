import os
import time
from collections.abc import AsyncIterable
from typing import Any, Literal, NamedTuple

import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

# Initialize a memory saver for agent state persistence
memory = MemorySaver()


class StockIndex(NamedTuple):
    name: str
    ticker: str  # Yahoo Finance symbol


# Each country's headline stock market index, as listed on Yahoo Finance.
# (Czech Republic is intentionally absent: Yahoo has no working symbol for the PX index.)
COUNTRY_INDICES: dict[str, StockIndex] = {
    "Germany": StockIndex("DAX", "^GDAXI"),
    "France": StockIndex("CAC 40", "^FCHI"),
    "Great Britain": StockIndex("FTSE 100", "^FTSE"),
    "Spain": StockIndex("IBEX 35", "^IBEX"),
    "Italy": StockIndex("FTSE MIB", "FTSEMIB.MI"),
    "Netherlands": StockIndex("AEX", "^AEX"),
    "Belgium": StockIndex("BEL 20", "^BFX"),
    "Sweden": StockIndex("OMX Stockholm 30", "^OMX"),
    "Norway": StockIndex("Oslo Bors Benchmark", "OSEBX.OL"),
    "Finland": StockIndex("OMX Helsinki 25", "^OMXH25"),
    "Denmark": StockIndex("OMX Copenhagen 25", "^OMXC25"),
    "Poland": StockIndex("WIG20", "WIG20.WA"),
    "Switzerland": StockIndex("SMI", "^SSMI"),
    "Austria": StockIndex("ATX", "^ATX"),
    "Portugal": StockIndex("PSI", "PSI20.LS"),
    "Greece": StockIndex("Athens General Index", "GD.AT"),
    "Ireland": StockIndex("ISEQ Overall", "^ISEQ"),
    "Hungary": StockIndex("BUX", "^BUX.BD"),
}
SUPPORTED_COUNTRIES = list(COUNTRY_INDICES)

# Common alternative spellings -> canonical key of COUNTRY_INDICES
COUNTRY_ALIASES = {
    "uk": "Great Britain",
    "u.k.": "Great Britain",
    "united kingdom": "Great Britain",
    "britain": "Great Britain",
    "england": "Great Britain",
    "gb": "Great Britain",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "deutschland": "Germany",
}

CACHE_TTL_SECONDS = 60
_quote_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def resolve_country(name: str) -> str | None:
    """Map free-form user input ('germany', 'UK', ' France ') to a supported country name."""
    key = name.strip().casefold()
    for country in COUNTRY_INDICES:
        if country.casefold() == key:
            return country
    return COUNTRY_ALIASES.get(key)


def fetch_index_quote(ticker: str) -> dict[str, Any]:
    """
    Fetch the latest close and previous close for a Yahoo Finance ticker.

    Results are cached for CACHE_TTL_SECONDS. Raises LookupError if Yahoo returns no
    usable data (unknown symbol, outage, ...).
    """
    cached = _quote_cache.get(ticker)
    if cached and time.monotonic() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    ticker_obj = yf.Ticker(ticker)
    # A 1-month window guarantees >= 2 trading days even over holidays / thin listings.
    history = ticker_obj.history(period="1mo", timeout=10)
    # Today's row can exist with NaN prices while the market is still opening - drop it.
    closes = history["Close"].dropna() if "Close" in history else history
    fast_info = getattr(ticker_obj, "fast_info", None) or {}

    if len(closes) >= 2:
        last, previous = float(closes.iloc[-1]), float(closes.iloc[-2])
    elif len(closes) == 1 and fast_info.get("previousClose"):
        # Some Yahoo symbols (e.g. WIG20, BUX) have a single price row and no history, but
        # still expose the previous close on the live quote.
        last, previous = float(closes.iloc[-1]), float(fast_info["previousClose"])
    else:
        raise LookupError(f"Yahoo Finance returned no usable price data for {ticker}")

    quote = {
        "last": last,
        "previous": previous,
        "change_pct": (last - previous) / previous * 100,
        "as_of": closes.index[-1].date().isoformat(),
        "currency": fast_info.get("currency") or "",
    }
    _quote_cache[ticker] = (time.monotonic(), quote)
    return quote


@tool
def get_stock_index_by_country(country: str) -> str:
    """
    Returns the latest value of a European country's main stock market index
    (e.g. DAX for Germany, CAC 40 for France) and its change versus the previous close.

    Args:
        country (str): The name of the European country (e.g., 'France').

    Returns:
        str: Index level, daily change and as-of date, or an error message.
    """
    resolved = resolve_country(country)
    if resolved is None:
        return (
            f"'{country}' is not a supported country. "
            f"Supported countries: {', '.join(SUPPORTED_COUNTRIES)}."
        )

    index = COUNTRY_INDICES[resolved]
    try:
        quote = fetch_index_quote(index.ticker)
    except Exception as e:
        return f"Could not fetch market data for {resolved} ({index.name}): {e}"

    return (
        f"{resolved} - {index.name} ({index.ticker}): "
        f"{quote['last']:,.2f} {quote['currency']}".rstrip()
        + f", {quote['change_pct']:+.2f}% vs previous close ({quote['previous']:,.2f}). "
        f"Latest value as of {quote['as_of']}; market data may be delayed."
    )


# Pydantic model for structuring agent responses
class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class StockInfoAgent:
    """StockInfoAgent - a specialized assistant for European stock market index data."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    # System prompt for the agent
    SYSTEM_INSTRUCTION = (
        "You are a stock market assistant specialized in the main stock market index of "
        "European countries (e.g. DAX for Germany, CAC 40 for France). "
        "Your sole purpose is to use the 'get_stock_index_by_country' tool to answer such queries. "
        "Call the tool once per country, and never invent or estimate index values yourself. "
        "Supported countries: " + ", ".join(SUPPORTED_COUNTRIES) + ". "
        "If the user asks about a country that is not supported or about unrelated topics, "
        "politely say you can only help with the stock market index of the supported European countries. "
        "Set response status to 'input_required' if the user needs to provide a valid country name. "
        "Set response status to 'error' if the tool fails or market data could not be retrieved. "
        "Set response status to 'completed' if the request is successfully processed."
    )

    def __init__(self, model=None):
        # Initialize the language model (injectable so tests can run without an API key)
        self.model = model or ChatGoogleGenerativeAI(
            # override with GEMINI_MODEL in .env when Google retires a model (read here, after .env is loaded)
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        )
        # Register the available tools
        self.tools = [get_stock_index_by_country]

        # Create the agent graph with the model, tools, and system prompt.
        # (The structured final answer is produced by _structured_response, not by LangGraph's
        # built-in response_format step, which ends the request on a model turn - Gemini 3 rejects that.)
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
        )

    FINAL_ANSWER_REQUEST = (
        "Using the conversation above, give your final answer as the structured response: "
        "a status (input_required, completed or error) and a message for the user."
    )

    async def _structured_response(self, config: RunnableConfig) -> ResponseFormat:
        """Ask the model for the final {status, message} answer, ending the request on a user turn."""
        messages = self.graph.get_state(config).values["messages"]
        structured_model = self.model.with_structured_output(ResponseFormat)
        result = await structured_model.ainvoke(
            [SystemMessage(self.SYSTEM_INSTRUCTION), *messages, HumanMessage(self.FINAL_ANSWER_REQUEST)]
        )
        return result

    async def invoke(self, query, context_id):
        # Prepare the config for the agent run
        config: RunnableConfig = {"configurable": {"thread_id": context_id}}
        await self.graph.ainvoke({"messages": [("user", query)]}, config)
        return self.get_agent_response(await self._structured_response(config))

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        """
        Asynchronously stream the agent's response to a query.
        """
        # Prepare the input and config for streaming
        inputs = {"messages": [("user", query)]}
        config: RunnableConfig = {"configurable": {"thread_id": context_id}}

        # astream (not stream) so the LLM / tool calls don't block the server's event loop
        async for item in self.graph.astream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Looking up market data...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing market data...",
                }

        # Yield the final agent response
        yield self.get_agent_response(await self._structured_response(config))

    def get_agent_response(self, structured_response: ResponseFormat | None):
        # Map the structured {status, message} answer to the dict the executor understands
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "error":
                # A real failure, not a request for input: the executor maps this to TaskState.failed
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "is_error": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        # Default fallback response if agent state is not as expected
        return {
            "is_task_complete": False,
            "require_user_input": False,
            "is_error": True,
            "content": (
                "We are unable to process your request at the moment. "
                "Please try again."
            ),
        }

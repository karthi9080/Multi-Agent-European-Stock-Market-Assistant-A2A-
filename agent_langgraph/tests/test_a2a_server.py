"""
End-to-end test of the A2A server: real A2A SDK server + client, real LangGraph ReAct loop and
real executor. Only the LLM is replaced by a scripted model (no API key needed) and Yahoo Finance
is faked, so the flow is deterministic.
"""
import uuid

import httpx
import pytest
from a2a.client import A2AClient
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MessageSendParams,
    SendMessageRequest,
    Task,
    TaskState,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.agent import ResponseFormat, StockInfoAgent
from app.agent_executor import StockInfoAgentExecutor


class ScriptedModel(BaseChatModel):
    """Fake LLM: optionally calls the stock tool once, then returns a fixed structured answer."""

    tool_country: str | None = None
    final: ResponseFormat = ResponseFormat(status="completed", message="ok")
    tool_calls_seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        already_called = any(isinstance(m, ToolMessage) for m in messages)
        if self.tool_country and not already_called:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "get_stock_index_by_country",
                             "args": {"country": self.tool_country}, "id": "call-1"}],
            )
        else:
            msg = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _messages: self.final)


def build_client(model, executor_agent=None):
    agent = executor_agent or StockInfoAgent(model=model)
    card = AgentCard(
        name="Stock Info Agent", description="test", url="http://test/", version="1.0.0",
        defaultInputModes=["text"], defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(id="s", name="s", description="s", tags=["t"])],
    )
    handler = DefaultRequestHandler(
        agent_executor=StockInfoAgentExecutor(agent), task_store=InMemoryTaskStore()
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return A2AClient(http, card, url="http://test/"), http


async def send(client, text, context_id=None, task_id=None) -> Task:
    message = {
        "role": "user", "parts": [{"type": "text", "text": text}],
        "messageId": str(uuid.uuid4()), "contextId": context_id or str(uuid.uuid4()),
    }
    if task_id:
        message["taskId"] = task_id
    req = SendMessageRequest(id=str(uuid.uuid4()), params=MessageSendParams.model_validate({"message": message}))
    resp = await client.send_message(req)
    return resp.root.result


async def test_completed_flow_returns_artifact_from_tool(fake_yf):
    calls = fake_yf(closes=[100.0, 103.0])
    model = ScriptedModel(tool_country="Germany",
                          final=ResponseFormat(status="completed", message="Germany DAX is up 3%"))
    client, http = build_client(model)
    task = await send(client, "How is Germany doing?")

    assert task.status.state == TaskState.completed
    assert calls[0] == "^GDAXI"  # the tool really ran through LangGraph
    assert task.artifacts[0].name == "stock_index_result"
    assert task.artifacts[0].parts[0].root.text == "Germany DAX is up 3%"
    await http.aclose()


async def test_input_required_for_unsupported_country(fake_yf):
    fake_yf()
    model = ScriptedModel(final=ResponseFormat(status="input_required", message="Which country?"))
    client, http = build_client(model)
    task = await send(client, "stock please")

    assert task.status.state == TaskState.input_required
    assert task.status.message.parts[0].root.text == "Which country?"
    await http.aclose()


async def test_agent_error_status_becomes_failed_task(fake_yf):
    """Issue: errors used to be reported as input_required."""
    fake_yf()
    model = ScriptedModel(final=ResponseFormat(status="error", message="Yahoo is down"))
    client, http = build_client(model)
    task = await send(client, "How is France?")

    assert task.status.state == TaskState.failed
    assert "Yahoo is down" in task.status.message.parts[0].root.text
    await http.aclose()


async def test_crash_inside_agent_becomes_failed_task():
    class Boom:
        async def stream(self, query, context_id):
            raise RuntimeError("llm exploded")
            yield  # pragma: no cover

    client, http = build_client(None, executor_agent=Boom())
    task = await send(client, "anything")

    assert task.status.state == TaskState.failed
    text = task.status.message.parts[0].root.text
    assert "Internal error" in text and "llm exploded" not in text  # internals are not leaked
    await http.aclose()


async def test_same_context_id_keeps_conversation_memory(fake_yf):
    """The LangGraph checkpointer is keyed by context_id, so a follow-up sees earlier turns."""
    fake_yf()
    model = ScriptedModel(final=ResponseFormat(status="completed", message="ok"))
    agent = StockInfoAgent(model=model)
    client, http = build_client(model, executor_agent=agent)
    ctx = str(uuid.uuid4())
    await send(client, "first question", context_id=ctx)
    await send(client, "second question", context_id=ctx)

    state = agent.graph.get_state({"configurable": {"thread_id": ctx}})
    human = [m.content for m in state.values["messages"] if m.type == "human"]
    assert human == ["first question", "second question"]
    await http.aclose()

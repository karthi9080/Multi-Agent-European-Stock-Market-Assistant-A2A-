from types import SimpleNamespace

import httpx
import pytest
from a2a.client import A2AClient

from host.agent import HostAgent, root_agent
from conftest import RECEIVED
import os


@pytest.fixture
async def host():
    RECEIVED.clear()
    return await HostAgent.create(os.environ["STOCK_AGENT_URL"])


def ctx():
    """Minimal stand-in for ADK's ToolContext (only .state is used)."""
    return SimpleNamespace(state={})


def test_root_agent_is_created_at_import_and_knows_remote_agent():
    assert root_agent is not None and root_agent.name == "Host_Agent"
    instruction = root_agent.instruction(SimpleNamespace())
    assert "Stock Info Agent" in instruction and "stock market" in instruction
    assert "friend" not in instruction.lower() and "in-place" not in instruction.lower()


async def test_completed_task_returns_text_from_artifact(host):
    out = await host.send_message("How is Germany?", ctx())
    assert out == {"status": "completed", "text": "ANSWER for: How is Germany?"}


async def test_context_id_is_stable_across_turns_and_task_id_is_not_reused(host):
    """Issue: IDs were regenerated on every call, so the remote agent never saw one conversation."""
    tool_ctx = ctx()
    await host.send_message("one", tool_ctx)
    assert tool_ctx.state["task_id"] is None  # finished task is forgotten, never re-used
    await host.send_message("two", tool_ctx)

    first, second = RECEIVED
    assert first["context_id"] == second["context_id"] == tool_ctx.state["context_id"]
    assert first["task_id"] != second["task_id"]  # each question is its own task in one conversation


async def test_different_conversations_get_different_context_ids(host):
    await host.send_message("a", ctx())
    await host.send_message("b", ctx())
    assert RECEIVED[0]["context_id"] != RECEIVED[1]["context_id"]


async def test_input_required_is_surfaced_and_followup_continues_the_same_task(host):
    """Issue: input_required has no artifacts, so the host used to return an empty list."""
    tool_ctx = ctx()
    out = await host.send_message("ask something vague", tool_ctx)
    assert out == {"status": "input_required", "text": "Which country?"}
    paused_task = tool_ctx.state["task_id"]
    assert paused_task

    out = await host.send_message("Germany", tool_ctx)
    assert RECEIVED[1]["task_id"] == paused_task  # follow-up continues the paused task
    assert out["status"] == "completed"
    assert tool_ctx.state["task_id"] is None  # and the finished task is not reused afterwards


async def test_failed_task_is_reported_as_failed_with_reason(host):
    out = await host.send_message("fail please", ctx())
    assert out == {"status": "failed", "text": "Yahoo is down"}


async def test_unreachable_remote_agent_returns_error_instead_of_crashing(host):
    dead_card = host.remote_agent_card.model_copy(update={"url": "http://127.0.0.1:1/"})
    dead = A2AClient(httpx.AsyncClient(timeout=2), dead_card)
    host.remote_agent_connection.agent_client = dead
    out = await host.send_message("How is Germany?", ctx())
    assert out["status"] == "error" and "Could not reach the Stock Info Agent" in out["text"]

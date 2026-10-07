"""
Starts a scripted fake "Stock Info Agent" as a real A2A HTTP server on a local port, and points the
host at it *before* host.agent is imported (host.agent connects to the remote agent at import time).
"""
import socket
import threading
import time
import os

import httpx
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TaskState, TextPart

RECEIVED: list[dict] = []  # (task_id, context_id, text) of every message the fake agent receives


def _parts(text):
    return [Part(root=TextPart(text=text))]


class ScriptedExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        RECEIVED.append({
            "task_id": context.message.task_id,
            "context_id": context.message.context_id,
            "text": text,
        })
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        if text.startswith("fail"):
            await updater.failed(message=updater.new_agent_message(_parts("Yahoo is down")))
        elif text.startswith("ask") and not context.current_task.status.state == TaskState.input_required \
                if context.current_task else text.startswith("ask"):
            await updater.update_status(
                TaskState.input_required, message=updater.new_agent_message(_parts("Which country?"))
            )
        else:
            await updater.add_artifact(_parts(f"ANSWER for: {text}"), name="stock_index_result")
            await updater.complete()

    async def cancel(self, context, event_queue):
        raise NotImplementedError


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_fake_remote_agent() -> str:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    card = AgentCard(
        name="Stock Info Agent", description="Fake agent for host tests", url=url + "/", version="1.0.0",
        defaultInputModes=["text"], defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(id="s", name="s", description="s", tags=["t"])],
    )
    handler = DefaultRequestHandler(agent_executor=ScriptedExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):  # wait until the agent card is served
        try:
            if httpx.get(url + "/.well-known/agent.json", timeout=1).status_code == 200:
                return url
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError("fake remote agent did not start")


# Must happen at import time of conftest, i.e. before any test module imports host.agent
os.environ["STOCK_AGENT_URL"] = _start_fake_remote_agent()

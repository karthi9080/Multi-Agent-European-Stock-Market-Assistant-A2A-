import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, AsyncIterable

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    SendMessageRequest,
    Task,
    TaskState,
    TextPart,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .remote_agent_connection import RemoteAgentConnection 

# Load environment variables from a .env file
load_dotenv()
# Allow nested event loops (useful for Jupyter and async environments)
nest_asyncio.apply()


class HostAgent:
    """The Host agent responsible for orchestrating European stock market index lookups."""

    def __init__(self, remote_agent_url: str): 
        # Store the remote agent's URL
        self.remote_agent_url = remote_agent_url
        # Will hold the connection to the remote agent (initialized later)
        self.remote_agent_connection: RemoteAgentConnection | None = None
        # Will hold the remote agent's card (initialized later)
        self.remote_agent_card: AgentCard | None = None
        # Create the internal agent instance
        self._agent = self.create_agent()
        # Set a static user ID for the host agent
        self._user_id = "host_agent"
        # Initialize the runner for managing sessions, memory, and artifacts
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self): 
        """
        Asynchronously initialize the remote agent connection and retrieve its card.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            card_resolver = A2ACardResolver(client, self.remote_agent_url)
            try:
                # Attempt to fetch the remote agent's card
                card = await card_resolver.get_agent_card()
                self.remote_agent_card = card # Store the single card
                # Establish the remote agent connection
                self.remote_agent_connection = RemoteAgentConnection( 
                    agent_card=card, agent_url=self.remote_agent_url
                )
            except httpx.ConnectError as e:
                print(
                    f"ERROR: Failed to get agent card from {self.remote_agent_url}: {e}. "
                    "Is the Stock Info Agent running? (cd agent_langgraph && uv run app/__main__.py)"
                )
                raise # Re-raise to ensure initialization fails if connection fails
            except Exception as e:
                print(f"ERROR: Failed to initialize connection for {self.remote_agent_url}: {e}")
                raise # Re-raise to ensure initialization fails if connection fails

    @classmethod
    async def create(cls, remote_agent_url: str): 
        """
        Asynchronously create and initialize a HostAgent instance.
        """
        instance = cls(remote_agent_url) 
        await instance._async_init_components()
        return instance

    def create_agent(self) -> Agent:
        """
        Create the internal Agent instance with its configuration and tools.
        """
        return Agent(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            name="Host_Agent",
            instruction=self.root_instruction,
            description="This Host agent answers questions about European stock market indices by delegating to the Stock Info Agent.",
            tools=[
                self.send_message,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """
        Generate the root instruction for the agent, including the remote agent's info.
        """
        # Prepare remote agent info for the instruction
        agent_info = (
            f'{{"name": "{self.remote_agent_card.name}", "description": "{self.remote_agent_card.description}"}}'
            if self.remote_agent_card
            else "No remote agent found"
        )
        return f"""
        **Role:** You are the Host Agent, a coordinator that answers questions about the stock market of European countries (the latest value of each country's main stock index, e.g. DAX for Germany). You gather the data by delegating to the Stock Info Agent.

        **Core Directives:**

        * **Understand Request:** Identify which European countries the user mentions.
        * **Delegate Requests:** Use the `send_message` tool to ask the Stock Info Agent, one country per call (e.g., "How is the stock market index in Germany doing?"). Never answer with market numbers yourself.
        * **Read the tool result:** `send_message` returns a `status` and a `text`:
            * `completed` - use `text` as the answer for that country.
            * `input_required` - the Stock Info Agent needs more information (e.g. a valid country). Relay its question to the user, and continue the conversation with their reply.
            * `failed` - the data could not be retrieved. Tell the user that this country failed and why; do not guess a value.
        * **Aggregate Results:** For several countries, combine the answers into one clear table or bullet list, and mention any country that failed or is unsupported.
        * **Handle Unsupported Requests:** If the query is not about the stock market of European countries, politely decline and clarify your purpose.
        * **Clarity & Brevity:** Keep your responses clear and easy to read. Remind the user that values are the latest close and may be delayed.

        **Today's Date (YYYY-MM-DD):** {datetime.now().strftime("%Y-%m-%d")}

        <Available Agent>
        {agent_info}
        </Available Agent>
        """

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Streams the agent's response to a given query.
        """
        # Retrieve or create a session for the user and session_id
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        # Prepare the user message content
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        if session is None:
            # If no session exists, create a new one
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        # Stream the agent's response as events
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                # If the event is the final response, extract and yield the content
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = "\n".join(
                        [p.text for p in event.content.parts if p.text]
                    )
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                # Otherwise, yield a thinking/progress update
                yield {
                    "is_task_complete": False,
                    "updates": "The host agent is thinking...",
                }

    @staticmethod
    def _text_of(parts) -> str:
        """Join the text of A2A message/artifact parts (ignores non-text parts)."""
        return "\n".join(
            p.root.text for p in parts or [] if isinstance(p.root, TextPart) and p.root.text
        )

    async def send_message(self, task: str, tool_context: ToolContext) -> dict[str, str]:
        """
        Sends a task to the Stock Info Agent.

        Returns a dict with:
          - status: "completed", "input_required", "failed" or "error"
          - text: the answer, the agent's follow-up question, or the failure reason
        """
        if not self.remote_agent_connection:
            raise ValueError("No remote agent connection established.")

        # Use the remote agent connection to send the message
        client = self.remote_agent_connection

        # context_id groups every turn of this conversation, so it is generated once and kept in
        # session state. task_id is only reused to *continue* a task the remote agent paused with
        # input_required; a finished task must not be re-used, so a new one is started instead.
        state = tool_context.state
        context_id = state.get("context_id") or str(uuid.uuid4())
        state["context_id"] = context_id
        task_id = state.get("task_id")
        message_id = str(uuid.uuid4())

        message: dict[str, Any] = {
            "role": "user",
            "parts": [{"type": "text", "text": task}],
            "messageId": message_id,
            "contextId": context_id,
        }
        if task_id:
            message["taskId"] = task_id

        # Create the SendMessageRequest object
        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate({"message": message})
        )
        # Send the message to the remote agent and await the response
        try:
            send_response = await client.send_message(message_request)
        except Exception as e:  # network failure, timeout, remote down, ...
            state["task_id"] = None
            return {"status": "error", "text": f"Could not reach the Stock Info Agent: {e}"}

        result = send_response.root
        if isinstance(result, JSONRPCErrorResponse):
            state["task_id"] = None
            return {"status": "error", "text": f"Stock Info Agent returned an error: {result.error.message}"}

        remote = result.result
        if isinstance(remote, Message):  # agent answered directly without creating a task
            state["task_id"] = None
            return {"status": "completed", "text": self._text_of(remote.parts)}

        assert isinstance(remote, Task)
        remote_state = remote.status.state
        status_text = self._text_of(remote.status.message.parts) if remote.status.message else ""

        if remote_state == TaskState.input_required:
            # Remember the paused task so the user's next reply continues it
            state["task_id"] = remote.id
            return {"status": "input_required", "text": status_text}

        state["task_id"] = None
        if remote_state == TaskState.completed:
            text = "\n".join(self._text_of(a.parts) for a in remote.artifacts or []) or status_text
            return {"status": "completed", "text": text}

        return {
            "status": "failed",
            "text": status_text or f"The Stock Info Agent ended the task in state '{remote_state.value}'.",
        }


def _get_initialized_host_agent_sync():
    """Synchronously creates and initializes the HostAgent."""

    async def _async_main():
        # URL of the Stock Info Agent (override with the STOCK_AGENT_URL environment variable)
        remote_agent_url = os.getenv("STOCK_AGENT_URL", "http://localhost:10004")

        print("initializing host agent")
        # Create and initialize the HostAgent instance asynchronously
        hosting_agent_instance = await HostAgent.create(
            remote_agent_url=remote_agent_url
        )
        print("HostAgent initialized")
        # Return the internal agent instance
        return hosting_agent_instance.create_agent()

    try:
        # Run the async initialization in a synchronous context
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize HostAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing HostAgent within an async function in your application."
            )
        else:
            raise


# Initialize the root agent at import time for immediate availability
root_agent = _get_initialized_host_agent_sync()
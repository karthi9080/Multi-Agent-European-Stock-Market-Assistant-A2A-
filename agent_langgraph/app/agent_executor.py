import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from app.agent import StockInfoAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To standardize and translate the outputs of any AI agent (your custom Agent) into a format the A2A server understands: TaskEvents, Messages, and TaskStatusUpdates.
# Your agent will not be A2A-compatible by default.
# Your agent's responses (raw LLM outputs, plain text, etc.) won’t produce the required A2A TaskEvents like:
# TaskStatusUpdateEvent
# TaskArtifactUpdateEvent
# Message
# The A2A Server won’t know:
# When your agent starts or finishes work
# What output to show to the user
# What artifacts to track


# As a result, your agent will not function in the A2A ecosystem (no updates, no display, no workflows).
# In A2A server default request handler uses agent_executor.execute such that we need this function to bridge the gap between server and agent


class StockInfoAgentExecutor(AgentExecutor):
    """AgentExecutor exposing the Stock Info Agent (European stock market indices) over A2A."""

    def __init__(self, agent: StockInfoAgent | None = None):
        # Initialize the StockInfoAgent instance (injectable for tests)
        self.agent = agent or StockInfoAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Validate that the context contains required identifiers and message
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        # Create a TaskUpdater to manage task state and events
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        # Extract the user query from the context
        query = context.get_user_input()
        try:
            # Stream the agent's response and update task state accordingly
            async for item in self.agent.stream(query, context.context_id):
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                parts = [Part(root=TextPart(text=item["content"]))]

                if item.get("is_error"):
                    # The agent could not fulfil the request: report a real failure, not "input required"
                    await updater.failed(message=updater.new_agent_message(parts))
                    break
                if not is_task_complete and not require_user_input:
                    # Agent is still working, update status
                    await updater.update_status(
                        TaskState.working,
                        message=updater.new_agent_message(parts),
                    )
                elif require_user_input:
                    # Agent requires more input from the user
                    await updater.update_status(
                        TaskState.input_required,
                        message=updater.new_agent_message(parts),
                    )
                    break
                else:
                    # Task is complete, add result as artifact and mark complete
                    await updater.add_artifact(
                        parts,
                        name="stock_index_result",
                    )
                    await updater.complete()
                    break

        except Exception:
            # Log the error and mark the task as failed so the caller sees a clean failure state
            # (details stay in the server log; the caller only gets a generic message)
            logger.exception("An error occurred while streaming the response")
            await updater.failed(
                message=updater.new_agent_message(
                    [Part(root=TextPart(text="Internal error while processing the request. Please try again later."))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Cancel is not supported for this agent executor
        raise ServerError(error=UnsupportedOperationError())

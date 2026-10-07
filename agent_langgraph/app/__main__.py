import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from app.agent import StockInfoAgent
from app.agent_executor import StockInfoAgentExecutor

# Load environment variables from a .env file
load_dotenv()

# Configure logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Starts Stock Info Agent server."""
    # Define the host and port for the server
    host = os.getenv("STOCK_AGENT_HOST", "localhost")
    port = int(os.getenv("STOCK_AGENT_PORT", "10004"))

    try:
        # Define the agent's capabilities (streaming only; push notifications are not used)
        capabilities = AgentCapabilities(streaming=True, pushNotifications=False)
        # Define the agent's skill (stock index lookup)
        skill = AgentSkill(
            id="stock_index_lookup",
            name="European Stock Index Lookup",
            description="Provides the latest value and daily change of the main stock market index of European countries (live data via Yahoo Finance)",
            tags=["stock", "market", "index", "europe", "finance"],
            examples=["How is the German stock market doing?", "What is the CAC 40 at?"],
        )

        # Create the agent card with metadata and supported content types
        agent_card = AgentCard(
            name="Stock Info Agent",
            description="Fetches the latest main stock market index value for supported European countries",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=StockInfoAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=StockInfoAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # Set up the request handler with the agent executor and task store
        request_handler = DefaultRequestHandler(
            agent_executor=StockInfoAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )

        # Build the Starlette application server with the agent card and handler
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        # Start the Uvicorn server
        uvicorn.run(server.build(), host=host, port=port)

    except Exception as e:
        # Log and exit if an error occurs during startup
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

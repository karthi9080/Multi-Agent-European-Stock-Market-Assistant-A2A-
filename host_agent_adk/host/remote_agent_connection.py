import httpx
from a2a.client import A2AClient
from a2a.types import AgentCard, SendMessageRequest, SendMessageResponse
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


class RemoteAgentConnection: 
    """A class to hold the connection to a single remote agent.""" 

    def __init__(self, agent_card: AgentCard, agent_url: str):
        # Print the agent card and URL for debugging purposes
        print(f"agent_card: {agent_card}")
        print(f"agent_url: {agent_url}")
        # Create an asynchronous HTTP client; 120 s because the remote agent makes two LLM calls
        # per request and Gemini can retry on temporary 503 'high demand' errors
        self._httpx_client = httpx.AsyncClient(timeout=120)
        # Initialize the A2AClient for communication with the remote agent
        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        # Store the agent card for later reference
        self.card = agent_card
        # Optionally store conversation metadata (not used in current logic)
        self.conversation_name = None
        self.conversation = None
        # Track any pending tasks (not used in current logic)
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        """
        Returns the stored agent card for this remote connection.
        """
        return self.card

    async def send_message(
        self, message_request: SendMessageRequest
    ) -> SendMessageResponse:
        """
        Sends a message to the remote agent using the A2AClient and returns the response.
        """
        return await self.agent_client.send_message(message_request)
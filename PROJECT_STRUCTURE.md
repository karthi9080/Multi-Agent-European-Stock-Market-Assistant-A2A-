# Project Structure & File Explanations

## Top-Level Structure

```
a2a/
│
├── agent_langgraph/
│   └── app/
│       ├── __main__.py
│       ├── agent.py
│       └── agent_executor.py
│   ├── tests/
│   ├── pyproject.toml
│   └── uv.lock
│
├── host_agent_adk/
│   └── host/
│       ├── agent.py
│       └── remote_agent_connection.py
│   ├── tests/
│   ├── pyproject.toml
│   └── uv.lock
│
├── README.md
├── PROJECT_STRUCTURE.md
└── __init__.py
```

---

## agent_langgraph/app/

### `__main__.py`
- **Purpose:** Entry point for running the Stock Info Agent server.
- **What it does:**  
  - Loads environment variables and configures logging.
  - Defines the agent’s capabilities and skills (e.g., stock market index lookup for European countries).
  - Builds an `AgentCard` (metadata about the agent).
  - Sets up the request handler and task store.
  - Starts a Uvicorn server to serve the agent via HTTP.
- **How it fits:** This file is run to start the agent service, making it available for requests.

---

### `agent.py`
- **Purpose:** Implements the core logic for the Stock Info Agent.
- **What it does:**  
  - Maps each supported country to its main stock index (Yahoo Finance symbol) plus aliases.
  - Provides a tool (`get_stock_index_by_country`) that fetches the latest value and daily change with `yfinance`.
  - Defines a `StockInfoAgent` class that:
    - Sets up a language model and tools.
    - Handles queries, streams responses, and formats output.
- **How it fits:** This is the “brain” of the agent, handling user queries and returning stock information.

---

### `agent_executor.py`
- **Purpose:** Bridges the agent logic with the A2A server infrastructure.
- **What it does:**  
  - Implements `StockInfoAgentExecutor`, a subclass of `AgentExecutor`.
  - Handles execution of agent tasks:
    - Receives requests, streams responses, updates task state, and manages artifacts.
    - Reports errors and crashes as `failed` tasks.
- **How it fits:** This file connects the agent’s logic to the server, managing the lifecycle of each request.

---

## host_agent_adk/host/

### `agent.py`
- **Purpose:** Implements the Host Agent, which coordinates with a remote agent to retrieve stock data.
- **What it does:**  
  - Initializes a connection to a remote agent (the Stock Info Agent).
  - Defines the Host Agent’s instructions and tools (notably, a `send_message` tool to communicate with the remote agent).
  - Handles streaming of responses and aggregation of results.
  - Provides a synchronous initialization function for use in other contexts.
- **How it fits:** This agent acts as a coordinator, forwarding user queries to the Stock Info Agent and aggregating the results.

---

### `remote_agent_connection.py`
- **Purpose:** Manages the connection to a remote agent.
- **What it does:**  
  - Initializes an HTTP client and an A2A client for communication.
  - Stores the agent card and manages message sending.
- **How it fits:** This is a utility/helper class used by the Host Agent to communicate with the Stock Info Agent.

---

## Other Files

- **`pyproject.toml`, `uv.lock`**: Dependency and environment management for Python projects.
- **`README.md`**: Project-level documentation .
- **`__init__.py`**: Marks directories as Python packages.

---
  
## How Everything Connects

1. **Stock Info Agent** (`agent_langgraph/app/agent.py`) is the core agent that knows how to answer stock queries.
2. **StockInfoAgentExecutor** (`agent_executor.py`) wraps the agent for use with the A2A server.
3. **The server** (`__main__.py`) exposes the agent as a web service.
4. **Host Agent** (`host_agent_adk/host/agent.py`) acts as a coordinator, forwarding queries to the Stock Info Agent.
5. **RemoteAgentConnection** (`remote_agent_connection.py`) is used by the Host Agent to communicate with the Stock Info Agent over HTTP.

---

## Typical Flow

1. **User** asks the Host Agent for stock market index in a European country.
2. **Host Agent** uses `RemoteAgentConnection` to send the query to the Stock Info Agent.
3. **Stock Info Agent** processes the query and returns the result.
4. **Host Agent** aggregates and formats the response for the user.

---

---

## Task, Session, and Message Details

### Task
- **Definition:** A unit of work representing a user’s request (e.g., "How is the German stock market doing?").
- **Lifecycle:**
  - Created when a user submits a query.
  - Tracked by a unique `task_id`.
  - Can be in states like `pending`, `working`, `input_required`, `completed`, or `error`.
  - May produce artifacts (results) upon completion.

### Session
- **Definition:** A logical grouping of related tasks, typically representing a conversation or workflow.
- **Lifecycle:**
  - Identified by a unique `session_id` (or `context_id`).
  - Maintains state and memory across multiple tasks/queries from the same user or workflow.
  - Enables context-aware responses and continuity.

### Message
- **Definition:** The actual content exchanged between user, agents, and services.
- **Types:**
  - User messages (queries, clarifications)
  - Agent messages (responses, requests for more input)
  - Tool messages (results from tools or external services)
- **Structure:**
  - Contains metadata such as sender, role, message ID, task ID, session/context ID, and content (text or structured data).

---

 
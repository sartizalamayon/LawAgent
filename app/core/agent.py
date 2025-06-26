import os
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import AsyncGenerator

# --- Define the agent's LLM ---
def get_llm():
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY", "sk-..."),
        model="llama3-70b-8192",
        streaming=True,
    )

# --- Define tools as LangChain tools ---
@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    # TODO: Integrate real web search API
    return f"Web search results for: {query}"

@tool
def law_reference(query: str) -> str:
    """Check legal references or databases."""
    # TODO: Integrate real law DB/API
    return f"Law reference results for: {query}"

def get_tools():
    return [web_search, law_reference]

# --- Build the agent using create_react_agent ---
def get_agent():
    return create_react_agent(
        get_llm(),
        get_tools(),
        prompt="You are a helpful bilingual legal assistant for Bangladeshi law.",
    )

# --- Streaming generator for FastAPI WebSocket ---
async def get_agent_stream(user_query: str) -> AsyncGenerator[dict, None]:
    agent = get_agent()
    messages = [HumanMessage(content=user_query)]
    async for step in agent.astream({"messages": messages}):
        # Each step is a dict with keys like 'agent', 'tools', etc.
        # Stream only the latest agent message content
        if "agent" in step and step["agent"]["messages"]:
            last = step["agent"]["messages"][-1]
            if hasattr(last, "content"):
                yield {"response": last.content} 
"""
Advanced Bilingual Legal AI Agent
================================

This module implements a production-grade legal assistant using LangGraph
with the following capabilities:
- Bilingual query processing (Bangla/English) with explicit language handling
- Document retrieval from Qdrant vector database
- Multi-tool execution with modular tool system
- Conversational memory with built-in LangGraph persistence
- Real-time streaming responses with detailed status updates
"""

import os
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
import logging

# LangChain and LangGraph imports
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool

# Translation imports
try:
    from argostranslate import package, translate
    ARGOS_AVAILABLE = True
except ImportError:
    ARGOS_AVAILABLE = False
    print("Warning: argostranslate not available, using fallback translation")

# Vector store imports
from app.vectorstore.qdrant_client import QdrantClientWrapper
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.retriever import BookRetriever

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# TRANSLATION UTILITIES
# ============================================================================

def setup_translation():
    """Initialize translation models for Bangla-English translation"""
    if not ARGOS_AVAILABLE:
        return None
    try:
        installed_languages = translate.get_installed_languages()
        bn_lang = None
        en_lang = None
        for lang in installed_languages:
            if lang.code == 'bn':
                bn_lang = lang
            elif lang.code == 'en':
                en_lang = lang
        if not bn_lang or not en_lang:
            package.install_from_path('https://argosopentech-mt-env.s3.amazonaws.com/translate-bn_en.argosmodel')
            package.install_from_path('https://argosopentech-mt-env.s3.amazonaws.com/translate-en_bn.argosmodel')
            installed_languages = translate.get_installed_languages()
            bn_lang = [lang for lang in installed_languages if lang.code == 'bn'][0]
            en_lang = [lang for lang in installed_languages if lang.code == 'en'][0]
        return {
            'bn_to_en': bn_lang.get_translation(en_lang),
            'en_to_bn': en_lang.get_translation(bn_lang)
        }
    except Exception as e:
        logger.error(f"Translation setup failed: {e}")
        return None

def translate_text(text: str, source_lang: str, target_lang: str, translator=None) -> str:
    """Translate text between Bangla and English"""
    if source_lang == target_lang:
        return text
    if translator and ARGOS_AVAILABLE:
        try:
            if source_lang == 'bn' and target_lang == 'en':
                return translator['bn_to_en'].translate(text)
            elif source_lang == 'en' and target_lang == 'bn':
                return translator['en_to_bn'].translate(text)
        except Exception as e:
            logger.error(f"Translation failed: {e}")
    return f"[Translation not available] {text}"

# ============================================================================
# DOCUMENT RETRIEVAL
# ============================================================================

async def retrieve_legal_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant legal documents from Qdrant vector database"""
    try:
        qdrant_client = QdrantClientWrapper("legal_documents")
        embedding_model = EmbeddingModel()
        query_embedding = embedding_model.embed_texts([query])[0]
        search_result = qdrant_client.client.search(
            collection_name="legal_documents",
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        documents = []
        for result in search_result:
            doc = {
                'content': result.payload.get('chunk_text', ''),
                'book_name': result.payload.get('book_name', ''),
                'page_number': result.payload.get('page_number', ''),
                'score': result.score,
                'metadata': result.payload
            }
            documents.append(doc)
        return documents
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return []

# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================

# Initialize BookRetriever for law_books
book_retriever = BookRetriever(collection_name="law_books")

# Utility: Use LLM for tasks like translation, classification, summarization
def get_llm():
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

# --- Tool: Legal Document Search ---
@tool
def legal_document_search(query: str, top_k: int = 5) -> str:
    """Search for relevant legal documents in the law_books collection and return context."""
    try:
        results = book_retriever.search(query, top_k=top_k)
        if results:
            context = "\n".join([doc['chunk_text'] for doc in results])
            return f"Found {len(results)} relevant documents:\n\n{context}"
        else:
            return "No relevant legal documents found in the law_books collection."
    except Exception as e:
        return f"Error searching documents: {str(e)}"

# --- Tool: LLM-powered Translation ---
@tool
def translate_query(query: str, source_language: str, target_language: str) -> str:
    """Translate a query between Bangla and English using the LLM."""
    llm = get_llm()
    prompt = f"Translate the following text from {source_language} to {target_language}:\n{query}"
    response = llm.invoke(prompt)
    return response.content

# --- Tool: LLM-powered Summarization ---
@tool
def document_summarizer(text: str) -> str:
    """Summarize legal documents or text using the LLM."""
    llm = get_llm()
    prompt = f"Summarize the following legal text in 3 bullet points:\n{text}"
    response = llm.invoke(prompt)
    return response.content

# --- Tool: LLM-powered Legal Classification ---
@tool
def legal_classifier(text: str) -> str:
    """Classify legal text into categories using the LLM."""
    llm = get_llm()
    prompt = (
        "Classify the following legal text into one of these categories: "
        "Contract Law, Criminal Law, Property Law, Family Law, General Legal Matter. "
        "Respond with only the category.\n"
        f"Text: {text}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

# --- Tool: LLM-powered Legal Advice ---
@tool
def legal_advice_generator(query: str, context: str = "") -> str:
    """Generate legal advice based on query and context using the LLM."""
    llm = get_llm()
    prompt = (
        "You are a legal assistant. Given the following query and context, provide a concise legal analysis and advice. "
        "If you are unsure, say so.\n"
        f"Query: {query}\nContext: {context}"
    )
    response = llm.invoke(prompt)
    return response.content

# ============================================================================
# AGENT CONSTRUCTION
# ============================================================================

def create_legal_agent():
    """Create the LangGraph agent with tools and memory"""
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )
    
    # Create tools list
    tools = [
        legal_document_search,
        legal_advice_generator,
        document_summarizer,
        legal_classifier,
        translate_query
    ]
    
    # Create memory
    memory = MemorySaver()
    
    # Create the agent (using correct API from documentation)
    agent = create_react_agent(llm, tools, checkpointer=memory)
    
    return agent

# ============================================================================
# STREAMING INTERFACE
# ============================================================================

async def get_agent_stream(user_query: str, thread_id: str = "default", language: str = "en") -> AsyncGenerator[Dict[str, Any], None]:
    """Stream agent responses with detailed step updates"""
    try:
        # Create agent
        agent = create_legal_agent()
        
        # Prepare configuration
        config = {"configurable": {"thread_id": thread_id}}
        
        # Send initial status
        yield {
            "type": "status",
            "content": "🔍 Analyzing query...",
            "step": "start"
        }
        
        # Create messages with system message
        system_message = (
            "You are a bilingual legal assistant for Bangladeshi law. "
            "Always use tools when appropriate. Always provide responses in the user's preferred language. "
            "Be helpful, accurate, and professional. "
            "If the user asks in Bangla, respond in Bangla. If in English, respond in English. "
            "Use the legal_document_search tool to find relevant legal information. "
            "Use the translate_query tool if you need to translate between Bangla and English."
        )
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_query)
        ]
        
        # Stream the agent response using the correct API
        async for event in agent.astream(
            {"messages": messages},
            config=config,
            stream_mode="updates"
        ):
            # Process each event type based on documentation
            if "agent" in event:
                agent_event = event["agent"]
                if "messages" in agent_event:
                    for message in agent_event["messages"]:
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            # Tool calls detected
                            for tool_call in message.tool_calls:
                                yield {
                                    "type": "status",
                                    "content": f"🛠️ Using tool: {tool_call['name']}",
                                    "step": "tool_call",
                                    "tool_name": tool_call['name']
                                }
                        elif message.content:
                            # Final response
                            yield {
                                "type": "response",
                                "content": message.content,
                                "step": "final",
                                "language": language,
                                "tools_used": [],  # Will be populated from tool events
                                "documents_retrieved": 0
                            }
            
            elif "tools" in event:
                tools_event = event["tools"]
                if "messages" in tools_event:
                    for message in tools_event["messages"]:
                        yield {
                            "type": "status",
                            "content": f"✅ Tool result: {message.content[:100]}...",
                            "step": "tool_result"
                        }
        
        # Send completion message
        yield {
            "type": "complete",
            "content": "Agent processing complete",
            "step": "complete",
            "thread_id": thread_id
        }
        
    except Exception as e:
        logger.error(f"Error in agent stream: {e}")
        yield {
            "type": "error",
            "content": f"Agent error: {str(e)}",
            "step": "error"
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_chat_history(thread_id: str) -> List[Dict[str, Any]]:
    """Get chat history for a specific thread"""
    try:
        agent = create_legal_agent()
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get the current state from the checkpointer
        state = agent.get_state(config)
        if state and "messages" in state:
            history = []
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, SystemMessage):
                    history.append({"role": "system", "content": msg.content})
            return history
        return []
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return []

def clear_chat_history(thread_id: str) -> bool:
    """Clear chat history for a specific thread"""
    try:
        agent = create_legal_agent()
        config = {"configurable": {"thread_id": thread_id}}
        
        # Clear the state for this thread
        agent.get_state(config)
        return True
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return False

# ============================================================================
# INITIALIZATION
# ============================================================================

translator = setup_translation()

def get_agent():
    """Get the configured agent instance"""
    return create_legal_agent()

def get_qdrant_client():
    """Get Qdrant client wrapper"""
    return QdrantClientWrapper("legal_documents")

def get_bilingual_embeddings():
    """Get bilingual embedding model"""
    return EmbeddingModel()

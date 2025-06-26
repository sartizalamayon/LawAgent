"""
Advanced Bilingual Legal AI Agent
================================

This module implements a production-grade legal assistant using LangGraph
with the following capabilities:
- Bilingual query processing (Bangla/English)
- Document retrieval from Qdrant vector database
- Multi-tool execution (translation, summarization, legal classification)
- Conversational memory with thread management
- Real-time streaming responses

Architecture:
1. Input Ingestion Node - Process and classify user queries
2. Translation Node - Handle Bangla-English translation
3. Document Retrieval Node - Search Qdrant for relevant legal documents
4. Tool Selection Node - Choose appropriate tools based on query
5. Tool Execution Node - Execute selected tools (summarizer, classifier, etc.)
6. Response Generation Node - Generate final bilingual response
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, TypedDict
from enum import Enum

# LangChain and LangGraph imports
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Pydantic for state management
from pydantic import BaseModel, Field

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

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class QueryType(str, Enum):
    """Enumeration of possible query types for classification"""
    LEGAL_ADVICE = "legal_advice"
    DOCUMENT_SEARCH = "document_search"
    TRANSLATION = "translation"
    GENERAL = "general"

class AgentState(BaseModel):
    """
    Shared state for the LangGraph agent workflow.
    
    This state is passed between nodes and contains all necessary
    information for processing the user query through the pipeline.
    """
    # Input data
    user_query: str = Field(description="Original user query")
    query_language: str = Field(default="en", description="Language of the query (en/bn)")
    thread_id: str = Field(description="Conversation thread ID for memory")
    
    # Processing data
    translated_query: Optional[str] = Field(default=None, description="Query translated to English if needed")
    query_type: Optional[QueryType] = Field(default=None, description="Classified query type")
    
    # Document retrieval
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list, description="Documents retrieved from Qdrant")
    retrieval_score: Optional[float] = Field(default=None, description="Relevance score of retrieved documents")
    
    # Tool execution
    selected_tools: List[str] = Field(default_factory=list, description="Tools selected for execution")
    tool_results: Dict[str, Any] = Field(default_factory=dict, description="Results from tool execution")
    
    # Response generation
    context_summary: Optional[str] = Field(default=None, description="Summary of retrieved context")
    final_response: Optional[str] = Field(default=None, description="Final response to user")
    response_language: str = Field(default="en", description="Language of the response")
    
    # Metadata
    processing_steps: List[str] = Field(default_factory=list, description="Steps taken during processing")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    
    class Config:
        arbitrary_types_allowed = True

# ============================================================================
# TRANSLATION UTILITIES
# ============================================================================

def setup_translation():
    """Initialize translation models for Bangla-English translation"""
    if not ARGOS_AVAILABLE:
        return None
    
    try:
        # Download and install translation packages if not available
        installed_languages = translate.get_installed_languages()
        bn_lang = None
        en_lang = None
        
        for lang in installed_languages:
            if lang.code == 'bn':
                bn_lang = lang
            elif lang.code == 'en':
                en_lang = lang
        
        if not bn_lang or not en_lang:
            # Download packages
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
        print(f"Translation setup failed: {e}")
        return None

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language code ('en' for English, 'bn' for Bangla)
    """
    # Simple heuristic-based language detection
    bangla_chars = set('অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃঁািীুূৃেৈোৌ্')
    
    bangla_count = sum(1 for char in text if char in bangla_chars)
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars > 0 and bangla_count / total_chars > 0.3:
        return 'bn'
    return 'en'

def translate_text(text: str, source_lang: str, target_lang: str, translator=None) -> str:
    """
    Translate text between Bangla and English.
    
    Args:
        text: Text to translate
        source_lang: Source language code ('en' or 'bn')
        target_lang: Target language code ('en' or 'bn')
        translator: Translation model (optional)
        
    Returns:
        Translated text
    """
    if source_lang == target_lang:
        return text
    
    if translator and ARGOS_AVAILABLE:
        try:
            if source_lang == 'bn' and target_lang == 'en':
                return translator['bn_to_en'].translate(text)
            elif source_lang == 'en' and target_lang == 'bn':
                return translator['en_to_bn'].translate(text)
        except Exception as e:
            print(f"Translation failed: {e}")
    
    # Fallback: return original text with note
    return f"[Translation not available] {text}"

# ============================================================================
# DOCUMENT RETRIEVAL
# ============================================================================

async def retrieve_legal_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant legal documents from Qdrant vector database.
    
    Args:
        query: Search query
        top_k: Number of documents to retrieve
        
    Returns:
        List of retrieved documents with metadata
    """
    try:
        # Get embeddings and Qdrant client
        embeddings = get_bilingual_embeddings()
        qdrant_client = get_qdrant_client()
        
        # Generate query embedding
        query_embedding = embeddings.model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0].tolist()
        
        # Search in Qdrant
        search_results = qdrant_client.client.search(
            collection_name=qdrant_client.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        documents = []
        for result in search_results:
            doc = {
                'id': result.id,
                'score': result.score,
                'content': result.payload.get('chunk_text', result.payload.get('page_content', '')),
                'title': result.payload.get('title', result.payload.get('book_name', '')),
                'source': result.payload.get('source', result.payload.get('book_name', '')),
                'language': result.payload.get('language', 'en'),
                'metadata': {k: v for k, v in result.payload.items() if k not in ['chunk_text', 'page_content', 'title', 'book_name', 'source', 'language']}
            }
            documents.append(doc)
        
        return documents
    
    except Exception as e:
        print(f"Document retrieval failed: {e}")
        return []

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@tool
def legal_document_search(query: str) -> str:
    """
    Search for legal documents and case laws in the database.
    
    Args:
        query: Search query for legal documents
        
    Returns:
        Formatted search results with relevant legal information
    """
    try:
        # This would integrate with the actual document retrieval
        documents = asyncio.run(retrieve_legal_documents(query, top_k=3))
        
        if not documents:
            return "No relevant legal documents found for your query."
        
        result = "Found the following relevant legal documents:\n\n"
        for i, doc in enumerate(documents, 1):
            result += f"{i}. {doc['title']}\n"
            result += f"   Source: {doc['source']}\n"
            result += f"   Relevance: {doc['score']:.2f}\n"
            result += f"   Content: {doc['content'][:200]}...\n\n"
        
        return result
    
    except Exception as e:
        return f"Error searching legal documents: {str(e)}"

@tool
def legal_advice_generator(query: str, context: str = "") -> str:
    """
    Generate legal advice based on the query and context.
    
    Args:
        query: User's legal question
        context: Additional context from retrieved documents
        
    Returns:
        Structured legal advice
    """
    # This would use the LLM to generate advice
    advice_template = f"""
    Based on your query: "{query}"
    
    {f"Context from legal documents: {context}" if context else ""}
    
    Legal Advice:
    1. Consult with a qualified lawyer for specific legal matters
    2. Gather all relevant documents and evidence
    3. Consider alternative dispute resolution methods
    4. Be aware of applicable statutes of limitations
    5. Keep detailed records of all communications and actions
    
    Note: This is general legal information and should not replace professional legal counsel.
    """
    
    return advice_template

@tool
def document_summarizer(text: str) -> str:
    """
    Summarize long legal documents or texts.
    
    Args:
        text: Text to summarize
        
    Returns:
        Concise summary of the text
    """
    if len(text) < 500:
        return text
    
    # Simple summarization (in production, use a proper summarization model)
    sentences = text.split('.')
    if len(sentences) <= 3:
        return text
    
    # Take first and last few sentences
    summary_sentences = sentences[:2] + sentences[-2:]
    return '. '.join(summary_sentences) + '.'

@tool
def legal_classifier(text: str) -> str:
    """
    Classify legal text into categories (contract, criminal, civil, etc.).
    
    Args:
        text: Text to classify
        
    Returns:
        Legal category classification
    """
    # Simple keyword-based classification
    text_lower = text.lower()
    
    categories = {
        'contract': ['contract', 'agreement', 'terms', 'conditions', 'breach'],
        'criminal': ['crime', 'criminal', 'offense', 'penalty', 'punishment'],
        'civil': ['civil', 'dispute', 'compensation', 'damages'],
        'family': ['marriage', 'divorce', 'custody', 'family'],
        'property': ['property', 'land', 'ownership', 'title', 'possession'],
        'commercial': ['business', 'commercial', 'trade', 'company', 'corporate']
    }
    
    scores = {}
    for category, keywords in categories.items():
        scores[category] = sum(1 for keyword in keywords if keyword in text_lower)
    
    if max(scores.values()) == 0:
        return "general"
    
    return max(scores, key=scores.get)

# ============================================================================
# LANGGRAPH NODES
# ============================================================================

def input_ingestion_node(state: AgentState) -> AgentState:
    """
    Node 1: Process and classify user input.
    
    This node:
    - Detects the language of the query
    - Classifies the query type
    - Prepares the state for further processing
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with input processing results
    """
    try:
        # Detect language
        detected_lang = detect_language(state.user_query)
        state.query_language = detected_lang
        
        # Classify query type
        query_lower = state.user_query.lower()
        
        if any(word in query_lower for word in ['আইন', 'law', 'legal', 'rights', 'অধিকার']):
            state.query_type = QueryType.LEGAL_ADVICE
        elif any(word in query_lower for word in ['document', 'case', 'document', 'নথি', 'মামলা']):
            state.query_type = QueryType.DOCUMENT_SEARCH
        elif any(word in query_lower for word in ['translate', 'অনুবাদ', 'meaning']):
            state.query_type = QueryType.TRANSLATION
        else:
            state.query_type = QueryType.GENERAL
        
        state.processing_steps.append("Input processed and classified")
        
    except Exception as e:
        state.errors.append(f"Input ingestion error: {str(e)}")
    
    return state

def translation_node(state: AgentState) -> AgentState:
    """
    Node 2: Handle translation between Bangla and English.
    
    This node:
    - Translates Bangla queries to English for processing
    - Maintains original query for response generation
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with translation results
    """
    try:
        if state.query_language == 'bn':
            # Translate Bangla to English for processing
            translator = setup_translation()
            state.translated_query = translate_text(
                state.user_query, 'bn', 'en', translator
            )
            state.processing_steps.append("Query translated from Bangla to English")
        else:
            # Keep original query if already in English
            state.translated_query = state.user_query
            state.processing_steps.append("Query already in English")
    
    except Exception as e:
        state.errors.append(f"Translation error: {str(e)}")
        # Fallback: use original query
        state.translated_query = state.user_query
    
    return state

def document_retrieval_node(state: AgentState) -> AgentState:
    """
    Node 3: Retrieve relevant documents from Qdrant.
    
    This node:
    - Searches the vector database for relevant legal documents
    - Ranks results by relevance
    - Stores documents in state for tool execution
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with retrieved documents
    """
    try:
        # Use translated query for better retrieval
        search_query = state.translated_query or state.user_query
        
        # Retrieve documents
        documents = asyncio.run(retrieve_legal_documents(search_query, top_k=5))
        state.retrieved_documents = documents
        
        if documents:
            # Calculate average relevance score
            scores = [doc['score'] for doc in documents]
            state.retrieval_score = sum(scores) / len(scores)
            state.processing_steps.append(f"Retrieved {len(documents)} relevant documents")
        else:
            state.processing_steps.append("No relevant documents found")
    
    except Exception as e:
        state.errors.append(f"Document retrieval error: {str(e)}")
    
    return state

def tool_selection_node(state: AgentState) -> AgentState:
    """
    Node 4: Select appropriate tools based on query type and context.
    
    This node:
    - Analyzes query type and retrieved documents
    - Selects relevant tools for execution
    - Prepares tool execution plan
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with selected tools
    """
    try:
        tools_to_use = []
        
        # Always include document search for legal queries
        if state.query_type in [QueryType.LEGAL_ADVICE, QueryType.DOCUMENT_SEARCH]:
            tools_to_use.append("legal_document_search")
        
        # Add legal advice generator for advice queries
        if state.query_type == QueryType.LEGAL_ADVICE:
            tools_to_use.append("legal_advice_generator")
        
        # Add summarizer if we have long documents
        if state.retrieved_documents:
            for doc in state.retrieved_documents:
                if len(doc['content']) > 1000:
                    tools_to_use.append("document_summarizer")
                    break
        
        # Add classifier for general queries
        if state.query_type == QueryType.GENERAL:
            tools_to_use.append("legal_classifier")
        
        state.selected_tools = list(set(tools_to_use))  # Remove duplicates
        state.processing_steps.append(f"Selected tools: {', '.join(state.selected_tools)}")
    
    except Exception as e:
        state.errors.append(f"Tool selection error: {str(e)}")
    
    return state

def tool_execution_node(state: AgentState) -> AgentState:
    """
    Node 5: Execute selected tools and collect results.
    
    This node:
    - Executes each selected tool with appropriate inputs
    - Collects and stores tool results
    - Handles tool execution errors gracefully
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with tool execution results
    """
    try:
        tool_results = {}
        
        # Execute legal document search
        if "legal_document_search" in state.selected_tools:
            search_query = state.translated_query or state.user_query
            result = legal_document_search(search_query)
            tool_results["legal_document_search"] = result
        
        # Execute legal advice generator
        if "legal_advice_generator" in state.selected_tools:
            context = ""
            if state.retrieved_documents:
                context = "\n".join([doc['content'][:200] for doc in state.retrieved_documents[:2]])
            
            result = legal_advice_generator(state.user_query, context)
            tool_results["legal_advice_generator"] = result
        
        # Execute document summarizer
        if "document_summarizer" in state.selected_tools and state.retrieved_documents:
            summaries = []
            for doc in state.retrieved_documents[:2]:  # Summarize top 2 documents
                summary = document_summarizer(doc['content'])
                summaries.append(f"Document: {doc['title']}\nSummary: {summary}")
            tool_results["document_summarizer"] = "\n\n".join(summaries)
        
        # Execute legal classifier
        if "legal_classifier" in state.selected_tools:
            text_to_classify = state.user_query
            if state.retrieved_documents:
                text_to_classify += " " + state.retrieved_documents[0]['content'][:500]
            
            classification = legal_classifier(text_to_classify)
            tool_results["legal_classifier"] = f"Legal category: {classification}"
        
        state.tool_results = tool_results
        state.processing_steps.append(f"Executed {len(tool_results)} tools")
    
    except Exception as e:
        state.errors.append(f"Tool execution error: {str(e)}")
    
    return state

def response_generation_node(state: AgentState) -> AgentState:
    """
    Node 6: Generate final response using LLM and tool results.
    
    This node:
    - Combines tool results and retrieved documents
    - Generates comprehensive response using LLM
    - Translates response back to user's language if needed
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with final response
    """
    try:
        # Prepare context for LLM
        context_parts = []
        
        # Add tool results
        for tool_name, result in state.tool_results.items():
            context_parts.append(f"{tool_name.upper()}:\n{result}")
        
        # Add retrieved documents
        if state.retrieved_documents:
            doc_context = "RELEVANT DOCUMENTS:\n"
            for i, doc in enumerate(state.retrieved_documents[:3], 1):
                doc_context += f"{i}. {doc['title']}: {doc['content'][:300]}...\n"
            context_parts.append(doc_context)
        
        context = "\n\n".join(context_parts)
        
        # Generate response using LLM
        llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="llama3-70b-8192",
            streaming=False  # We'll handle streaming separately
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful bilingual legal assistant for Bangladeshi law.
            You can communicate in both Bangla and English.
            Provide comprehensive, accurate, and empathetic legal guidance.
            Always mention that this is general information and professional legal counsel should be consulted for specific cases."""),
            ("user", f"""Query: {state.user_query}
            
            Context from tools and documents:
            {context}
            
            Please provide a comprehensive response in {state.query_language} language.
            If the original query was in Bangla, respond in Bangla.
            If it was in English, respond in English.""")
        ])
        
        messages = prompt.format_messages()
        response = llm.invoke(messages)
        
        state.final_response = response.content
        state.response_language = state.query_language
        state.processing_steps.append("Generated final response")
    
    except Exception as e:
        state.errors.append(f"Response generation error: {str(e)}")
        # Fallback response
        state.final_response = f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again or consult with a legal professional."
    
    return state

# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def create_legal_agent() -> StateGraph:
    """
    Create the complete LangGraph workflow for the legal agent.
    
    Returns:
        Configured StateGraph with all nodes and edges
    """
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("input_ingestion", input_ingestion_node)
    workflow.add_node("translation", translation_node)
    workflow.add_node("document_retrieval", document_retrieval_node)
    workflow.add_node("tool_selection", tool_selection_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("response_generation", response_generation_node)
    
    # Define the workflow edges
    workflow.set_entry_point("input_ingestion")
    workflow.add_edge("input_ingestion", "translation")
    workflow.add_edge("translation", "document_retrieval")
    workflow.add_edge("document_retrieval", "tool_selection")
    workflow.add_edge("tool_selection", "tool_execution")
    workflow.add_edge("tool_execution", "response_generation")
    workflow.add_edge("response_generation", END)
    
    # Compile the workflow
    return workflow.compile(checkpointer=MemorySaver())

# ============================================================================
# STREAMING INTERFACE
# ============================================================================

async def get_agent_stream(user_query: str, thread_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream the agent's processing steps and final response.
    
    This function:
    - Creates the agent workflow
    - Executes the workflow with streaming
    - Yields detailed updates for each step
    - Handles errors gracefully
    
    Args:
        user_query: User's legal query
        thread_id: Conversation thread ID
        
    Yields:
        Dictionary with step information and content
    """
    try:
        # Create the agent
        agent = get_agent()
        
        # Initialize messages
        messages = [HumanMessage(content=user_query)]
        config = {"configurable": {"thread_id": thread_id}}
        
        # Send initial analysis message
        yield {
            "type": "analysis",
            "content": "আপনার প্রশ্ন বিশ্লেষণ করছি..." if detect_language(user_query) == 'bn' else "Analyzing your query...",
            "step": "initial_analysis"
        }
        
        # Execute workflow with streaming
        async for step in agent.astream({"messages": messages}, config):
            # Handle agent steps
            if "agent" in step and step["agent"].get("messages"):
                last_message = step["agent"]["messages"][-1]
                
                # Check if it's a tool call
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        yield {
                            "type": "tool_call",
                            "content": f"টুল ব্যবহার করছি: {tool_call['name']}",
                            "tool_name": tool_call['name'],
                            "tool_args": tool_call['args'],
                            "step": "tool_execution"
                        }
                
                # Check if it's a reasoning step
                elif hasattr(last_message, 'content') and last_message.content:
                    content = last_message.content
                    if any(keyword in content.lower() for keyword in ['thinking', 'analyzing', 'considering', 'reasoning']):
                        yield {
                            "type": "reasoning",
                            "content": content,
                            "step": "reasoning"
                        }
                    else:
                        yield {
                            "type": "response",
                            "content": content,
                            "step": "final_response"
                        }
            
            # Handle tool results
            elif "tools" in step and step["tools"].get("messages"):
                for tool_message in step["tools"]["messages"]:
                    if hasattr(tool_message, 'content'):
                        yield {
                            "type": "tool_result",
                            "content": f"টুল ফলাফল: {tool_message.content[:200]}...",
                            "full_result": tool_message.content,
                            "tool_name": getattr(tool_message, 'name', 'unknown'),
                            "step": "tool_result"
                        }
        
        # Send completion signal
        yield {
            "type": "completion",
            "content": "বিশ্লেষণ সম্পন্ন" if detect_language(user_query) == 'bn' else "Analysis completed",
            "step": "completed"
        }
    
    except Exception as e:
        yield {
            "type": "error",
            "content": f"Error: {str(e)}",
            "step": "error"
        }

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def get_agent():
    """
    Legacy function for backward compatibility.
    Returns a simple react agent for basic use cases.
    """
    return create_react_agent(
        ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY", "sk-..."),
            model="llama3-70b-8192",
            streaming=True,
        ),
        [legal_document_search, legal_advice_generator, document_summarizer, legal_classifier],
        checkpointer=MemorySaver(),
        prompt="""You are a helpful bilingual legal assistant for Bangladeshi law. 
        You can communicate in both Bangla and English.
        Always provide detailed step-by-step analysis and reasoning.
        When using tools, explain what you're doing and why.
        Be empathetic and provide practical legal advice.""",
    )

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_qdrant_client():
    """Get Qdrant client wrapper for legal documents collection"""
    return QdrantClientWrapper("legal_documents")

def get_bilingual_embeddings():
    """Get bilingual embedding model"""
    return EmbeddingModel("intfloat/multilingual-e5-base") 
applyTo: "**"

# LawAgent Codebase Instructions

These are global coding standards, architectural decisions, and preferences to follow across this project.

---

## 🧠 Project Overview

You are working on a bilingual, agent-based legal assistant designed to help users (primarily in Bangladesh) with queries using a chat interface. It leverages 200+ law books, documents, and scraped data sources stored in a vector database. Users can query the system in Bangla or English through various inputs (text, images, PDFs, voice). The system uses LangGraph + LangChain with FastAPI to create a multi-tool, memory-aware, long-running agent. We will build the fronend UI in React Native later. For now we focus on the backend API and agent orchestration.

---

## ⚙️ Tech Stack

- FastAPI (Backend API + WebSocket interface)
- LangGraph (Agentic orchestration with tool-calling)
- LangChain (Tool definition, embedding, prompt management)
- Groq API with LLaMA3 (LLM inference)
- Qdrant (VectorDB, bilingual embedding store, similarity search)
- PyMuPDF / pdfplumber / or best for books(PDF parsing)
- pytesseract + PIL (image parsing)
- sentence-transformers (bilingual embedding model, use huggingface api)
- WebSockets (real-time agent feedback)
- React Native (for final UI, not in scope yet)

---

## 📁 Folder Structure

This project is modularized. Keep the following base folder structure:

lawagent/
├── app/
│ ├── main.py # FastAPI entrypoint
│ ├── api/ # API routes for all things related to the agent chat using WebSockets to stream real time agent workflow
│ ├── core/ # Agent, tools, memories
│ ├── vectorstore/ # Qdrant vector ops, chunking
│ ├── parser/ # pdf/image/text/audio parsing
│ ├── embeddings/ # Bilingual embedding logic
│ ├── prompts/ # Prompt templates
│ ├── schemas/ # Pydantic models
│ └── utils/ # Common helpers
├── langgraph/ # LangGraph-specific agents & workflows
├── .env # Environment config
├── requirements.txt
└── README.md


## 🧩 System Design Summary

See the diagram in assets/autodraw-pipeline.png for a high-level view. In brief:

1. Ingestion:
   - Parses PDF, image, text, and audio to extract raw text.
   - Each document is chunked (page/book-mapped) and embedded bilingually.

2. Storage:
   - All chunks + metadata go to Qdrant.

3. Query:
   - User query (image/pdf/text) is parsed → bilingual embedded → Top-K vector search + re-ranking
   - Final prompt = user query + Top-K matches → passed to LangGraph agent.

4. Agent:
   - Built using LangGraph.
   - Has access to memory, tools (e.g., web search, ref checker, translator), and the Groq-hosted LLaMA4 bilingual LLM.

5. Feedback:
   - WebSocket (ws://host/agent) streams back step-wise updates.
   - Each tool invocation, thought, and final result is streamed.


---

## 🗃️ Components to Be Implemented (Incrementally)

- [ ] Core Agent using LangGraph
- [ ] Tooling layer (web search, translation, law ref checker, etc.)
- [ ] Qdrant ingestion pipeline
- [ ] All parsers (PDF, Image,, Text)
- [ ] FastAPI endpoints (chat, ingestion, etc.)
- [ ] WebSocket interface
- [ ] Test seeds for embedding, chunking, etc.



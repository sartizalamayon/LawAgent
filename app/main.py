from fastapi import FastAPI
from app.api import ingest, trigger
#from app.api import retriever  # Disabled for agent MVP
from app.api.agent_ws import router as agent_router

app = FastAPI()

app.include_router(ingest.router, prefix="/api")
app.include_router(trigger.router, prefix="/api")
#app.include_router(retriever.router, prefix="/api")  # Disabled for agent MVP
app.include_router(agent_router)  # No prefix, /agent/chat

@app.get("/")
async def root():
    return {"message": "Welcome to LawAgent"}

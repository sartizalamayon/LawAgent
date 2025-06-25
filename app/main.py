from fastapi import FastAPI
from app.api import ingest, trigger, retriever

app = FastAPI()

app.include_router(ingest.router, prefix="/api")
app.include_router(trigger.router, prefix="/api")
app.include_router(retriever.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to LawAgent"}

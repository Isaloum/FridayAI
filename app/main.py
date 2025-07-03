# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from FridayOS import Dispatcher
from dotenv import load_dotenv
load_dotenv()               # ← reads .env into os.environ

import os
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

class UserRequest(BaseModel):
    user_id: str
    text: str

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/dispatch")
async def dispatch(req: UserRequest):
    return Dispatcher.process_request(req.text, req.user_id)

"""
A minimalist FastAPI server that does nothing except say “I’m alive”.
Celery runs in the background; this stays in the foreground for Render.
"""
import os
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"status": "up"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
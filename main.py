from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import secrets
from engine.generator import generate_video
from pydantic import BaseModel
import asyncio
from engine.db_operations import get_status_by_token
from concurrent.futures import ProcessPoolExecutor


app = FastAPI()
executor = ProcessPoolExecutor()

origins = [
    "http://localhost:3000"  # Allow frontend origin
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


def generate_wikivid_token(length: int = 10) -> str:
    # Generate a random alphanumeric token
    # since each byte produces 2 hex characters
    token = secrets.token_hex(length // 2)
    return token.upper()


@app.get("/video-status/{token}")
async def check_status(token: str):
    # Check the status of the video generation request
    # Return the status and video URL if available
    return get_status_by_token(token)


@app.get("/generate-token")
async def generate_token():
    # Prefix the generated token with 'WV' for wikivid
    return {"token": "WV" + generate_wikivid_token()}


@app.post("/generate-video")
async def generate_video_req(background_tasks: BackgroundTasks, url: str = Form(...), token: str = Form(...)):
    print(f"URL received: {url}")
    print(f"Token received: {token}")
    background_tasks.add_task(generate_video, url, token)
    return {"message": "Video generation started", "token": token}

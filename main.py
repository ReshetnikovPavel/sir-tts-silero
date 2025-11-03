import os
import tempfile

from silero import silero_tts
from fastapi import FastAPI, HTTPException
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

import torchaudio


app = FastAPI()


class Request(BaseModel):
    input: str
    model: str
    voice: str
    instructions: str | None = None
    response_format: str = "wav"
    speed: float = 1.0


SAMPLE_RATE = 48000

MODEL, _ = silero_tts(language="ru", speaker="v5_ru")


@app.post("/v1/audio/speech")
async def create_speech(request: Request, bg_tasks: BackgroundTasks) -> FileResponse:
    input = request.input
    model = request.model
    voice = request.voice
    instructions = request.instructions
    response_format = request.response_format
    speed = request.speed

    voices = ["aidar", "baya", "kseniya", "xenia", "eugene"]
    if voice not in voices:
        raise HTTPException(
            status_code=400,
            detail=f"Voice `{voice}` is not supported. Possible values: {voices}",
        )

    if response_format != "wav":
        raise HTTPException(
            status_code=400,
            detail=f"Response format `{response_format}` is not supported",
        )

    output = MODEL.apply_tts(text=input, sample_rate=SAMPLE_RATE, speaker=voice)

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    torchaudio.save(uri=tmp_path, src=output, sample_rate=SAMPLE_RATE)
    bg_tasks.add_task(os.remove, tmp_path)
    return FileResponse(tmp_path, background=bg_tasks)
